import scipy as sp
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt



class ASM_Model():

    def __init__(self, data, author_num, iterations=3000, prior='uniform', burn_in=250):
        #[RA, DUP, EXT, DEV, ETF | CS, MNR, BST, RFR | type | author | ture_type], sorted by author id
        self.data = data
        self.iterations = iterations
        self.author_num = author_num
        self.data_num = len(data)
        self.burn_in = burn_in

        # raw label and true label
        self.raw_label = data[:, -3]
        self.raw_precision = np.sum(self.raw_label == data[:, -1])/self.data_num
        # print(self.raw_precision)

        # split spam reviews and non-spam reviews

        self.spam_num = int(np.sum(data[:, -3]))
        self.non_spam_num = self.data_num - self.spam_num
        copy_data = self.data[self.data[:, -3].argsort()]
        self.non_spam_reviews = copy_data[:self.non_spam_num, :]
        self.spam_reviews = copy_data[self.non_spam_num:, :]
        del copy_data

        # init hyper parameter
        if prior == 'uniform':
            # init theta's parameter
            self.gamma_ra_n_a = 1
            self.gamma_ra_n_b = 1
            self.gamma_ra_s_a = 1
            self.gamma_ra_s_b = 1

            self.gamma_dup_n_a = 1
            self.gamma_dup_n_b = 1
            self.gamma_dup_s_a = 1
            self.gamma_dup_s_b = 1

            self.gamma_ext_n_a = 1
            self.gamma_ext_n_b = 1
            self.gamma_ext_s_a = 1
            self.gamma_ext_s_b = 1

            self.gamma_dev_n_a = 1
            self.gamma_dev_n_b = 1
            self.gamma_dev_s_a = 1
            self.gamma_dev_s_b = 1

            self.gamma_etf_n_a = 1
            self.gamma_etf_n_b = 1
            self.gamma_etf_s_a = 1
            self.gamma_etf_s_b = 1

            # init author spamicity
            self.authors_alpha_a = np.ones(shape=(author_num,))
            self.authors_alpha_b = np.ones(shape=(author_num,))

    # count spam reviews
    def __count_spam(self):
        self.spam_num = int(np.sum(self.data[:, -3]))
        self.non_spam_num = self.data_num - self.spam_num

    # split spam review and non-spam review
    def __split_data(self):
        # [RA, DUP, EXT, DEV, ETF | CS, MNR, BST, RFR | type | author | ture_type], sorted by author id
        copy_data = self.data[self.data[:, -3].argsort()]
        self.non_spam_reviews = copy_data[:self.non_spam_num, :]
        self.spam_reviews = copy_data[self.non_spam_num:, :]
        # del copy_data


    def extract_author_info(self):
        # 最后一列，按作者序号排序 (传入时即已经有序，0~author_num)
        # self.data = self.data[self.data[:, -1].argsort()]
        authors = self.data[:, -2]
        # 每个作者的评论数
        self.authors_review_num = np.array([])
        for i in range(self.author_num):
            self.authors_review_num = np.append(self.authors_review_num, np.sum(authors == i))

        # 统计作者的垃圾评论数
        self.__count_author_spam()

    def __count_author_spam(self):
        start = 0
        # copy_data = self.data[self.data[:, -1].argsort()]
        self.author_spam = np.array([],dtype='int')
        for num in self.authors_review_num:
            reviews = self.data[start:start+int(num), -3]
            spam_num = np.sum(reviews)
            self.author_spam = np.append(self.author_spam, spam_num)
            start += int(num)

    def __update_author_behavior(self, author, review):
        if review[-3] == 1:
            self.author_spam[author] -= 1
        else:
            self.author_spam[author] += 1


    # sample stable data using  gibbs sample
    def gibbs_sampler(self):

        for iteration in range(self.iterations):
            details = []
            review_index = 0
            change_count = 0
            one_count = 0
            # for author 1 to A
            for author in range(self.author_num):
                # for review 1 to Ra
                for index in range(int(self.authors_review_num[author])):
                    # assign review cluster
                    # self.data[review_index][-2] = 'sample result'
                    assignment = self.__conditional_probability(self.data[review_index], iteration, details)
                    if assignment == 1:
                        one_count += 1
                    if assignment != self.data[review_index][-3]:
                        change_count += 1
                        # update author behavior
                        self.__update_author_behavior(author=author, review=self.data[review_index])
                        # update n_k_f , spam_num, non-spam_num
                        self.__update_disperse_samples(self.data[review_index])
                        # update data cluster
                        self.data[review_index][-3] = assignment

                        # test
                        # self.__split_data()
                        # self.__update_continuous_samples()
                    review_index += 1

            # record update parameters details
            if change_count > self.data_num//10:
                with open('./log1.txt', 'w') as f:
                    f.writelines(details)

            print('one_count', one_count)
            # print('change_count', change_count)
            if iteration > self.burn_in:
            # if iteration == self.iterations-2:
                # 重新分割数据，update spam_reviews, non-spam_reviews
                print(self.non_spam_num)
                self.__split_data()
                # update psi
                s, n = self.__collect_parameters()
                print('-----------------before update author feature-----------------')
                print('psi_spam', s)
                print('psi_non_spam', n)
                self.__update_continuous_samples()
                s, n = self.__collect_parameters()
                print('-----------------after update author feature-----------------')
                print('psi_spam', s)
                print('psi_non_spam', n)
                print(s)
                print(n)
                # exit(0)
            # output sample processing
            print('INFO:-----'+str(iteration)+'/'+str(self.iterations)+' ,准确率：'+str(self.Precision())+'--------')

    '''
    smaple formula for review i:
    P(𝜋i|features...) ∝ (𝑛𝑎,𝑘¬i + 𝛼𝑘,i)/(𝑛𝑎+𝛼𝑠,𝑎+𝛼𝑛,a)¬i
                        * ∏𝑓∈{DUP, EXT, DEV, ETF, RA}{𝑔(𝑓, 𝑘, 𝑥𝑎,r,f)}
                        * ∏𝑓∈{CS, MNR, BST, RFR}{(𝑝(𝑦𝑎,𝑟,𝑓|𝜓𝜋𝑖,f}
    '''
    def __conditional_probability(self, review, iteration, details):
        # review features
        ra, dup, ext, dev, etf, cs, mnr, bst, rfr, cluster, author, true_label = review

        # count
        spam_count = cluster
        non_spamcount = 0 if cluster else 1
        author = int(author)

        # (𝑛𝑎,s¬i + 𝛼s,a)/(𝑛𝑎+𝛼𝑠,𝑎+𝛼𝑛,a)¬i
        author_spamicity_s = ((self.author_spam[author] - cluster) + self.authors_alpha_a[author]) \
                / (self.authors_review_num[author] - 1 + self.authors_alpha_a[author] + self.authors_alpha_b[author])

        # (𝑛𝑎,non-spam¬i + 𝛼non-spam,a)/(𝑛𝑎+𝛼𝑠,𝑎+𝛼𝑛,a)¬i
        author_spamicity_n = (self.authors_review_num[author] - self.author_spam[author] + self.authors_alpha_b[author]
                - non_spamcount) / (self.authors_review_num[author] - 1 + self.authors_alpha_a[author] +
                                            self.authors_alpha_b[author])

        # ∏𝑓∈{RA, DUP, EXT, DEV, ETF}{𝑔(𝑓, 𝑘, 𝑥𝑎,r,f)}
        review_factor_s = (((self.n_ra_s_1 + self.gamma_ra_s_a) if ra else (self.n_ra_s_0 + self.gamma_ra_s_b)) -
            cluster) / (self.spam_num + self.gamma_ra_s_a + self.gamma_ra_s_b - cluster) * (
            ((self.n_dup_s_1 + self.gamma_dup_s_a) if dup else (self.n_dup_s_0 + self.gamma_dup_s_b)) - cluster) / (
                self.spam_num + self.gamma_dup_s_a + self.gamma_dup_s_b - cluster) * (
            ((self.n_ext_s_1 + self.gamma_ext_s_a) if ext else (self.n_ext_s_0 + self.gamma_ext_s_b)) - cluster) / (
                self.spam_num + self.gamma_ext_s_a + self.gamma_ext_s_b - cluster) * (
            ((self.n_dev_s_1 + self.gamma_dev_s_a) if dev else (self.n_dev_s_0 + self.gamma_dev_s_b)) - cluster) / (
                self.spam_num + self.gamma_dev_s_a + self.gamma_dev_s_b - cluster) * (
            ((self.n_etf_s_1 + self.gamma_etf_s_a) if etf else (self.n_etf_s_0 + self.gamma_etf_s_b)) - cluster) / (
                self.spam_num + self.gamma_etf_s_a + self.gamma_etf_s_b - cluster)

        # record factor to debug
        review_factor_s_compoments = {}
        review_factor_s_compoments['n_ra_s'] = self.n_ra_s_1 if ra else self.n_ra_s_0
        review_factor_s_compoments['n_dup_s'] = self.n_dup_s_1 if dup else self.n_dup_s_0
        review_factor_s_compoments['n_ext_s'] = self.n_ext_s_1 if ext else self.n_ext_s_0
        review_factor_s_compoments['n_dev_s'] = self.n_dev_s_1 if dev else self.n_dev_s_0
        review_factor_s_compoments['n_etf_s'] = self.n_etf_s_1 if etf else self.n_etf_s_0

        #
        review_factor_n = (((self.n_ra_n_1 + self.gamma_ra_n_a) if ra else (self.n_ra_n_0 + self.gamma_ra_n_b)) -
            non_spamcount) / (self.non_spam_num + self.gamma_ra_n_a + self.gamma_ra_n_b - non_spamcount) * (
            ((self.n_dup_n_1 + self.gamma_dup_n_a) if dup else (self.n_dup_n_0 + self.gamma_dup_n_b)) - non_spamcount) / (
                self.non_spam_num + self.gamma_dup_n_a + self.gamma_dup_n_b - non_spamcount) * (
            ((self.n_ext_n_1 + self.gamma_ext_n_a) if ext else (self.n_ext_n_0 + self.gamma_ext_n_b)) - non_spamcount) / (
                self.non_spam_num + self.gamma_ext_n_a + self.gamma_ext_n_b - non_spamcount) * (
            ((self.n_dev_n_1 + self.gamma_dev_n_a) if dev else (self.n_dev_n_0 + self.gamma_dev_n_b)) - non_spamcount) / (
                self.non_spam_num + self.gamma_dev_n_a + self.gamma_dev_n_b - non_spamcount) * (
            ((self.n_etf_n_1 + self.gamma_etf_n_a) if etf else (self.n_etf_n_0 + self.gamma_etf_n_b)) - non_spamcount) / (
                self.non_spam_num + self.gamma_etf_n_a + self.gamma_etf_n_b - non_spamcount)

        # debug
        if review_factor_s < 0 :
            if (((self.n_ra_s_1 + self.gamma_ra_s_a) if ra else (self.n_ra_s_0 + self.gamma_ra_s_b)) - cluster) / (
                self.spam_num + self.gamma_ra_s_a + self.gamma_ra_s_b - cluster) < 0:
                print('a')

            if (((self.n_dup_s_1 + self.gamma_dup_s_a) if dup else (self.n_dup_s_0 + self.gamma_dup_s_b)) - cluster) / (
                self.spam_num + self.gamma_dup_s_a + self.gamma_dup_s_b - cluster) <0 :
                print('b')

            if (((self.n_ext_s_1 + self.gamma_ext_s_a) if ext else (self.n_ext_s_0 + self.gamma_ext_s_b)) - cluster) / (
                self.spam_num + self.gamma_ext_s_a + self.gamma_ext_s_b - cluster) < 0:
                print('c')
            if (((self.n_dev_s_1 + self.gamma_dev_s_a) if dev else (self.n_dev_s_0 + self.gamma_dev_s_b)) - cluster) / (
                self.spam_num + self.gamma_dev_s_a + self.gamma_dev_s_b - cluster) < 0:
                print('d')

            if (((self.n_etf_s_1 + self.gamma_etf_s_a) if etf else (self.n_etf_s_0 + self.gamma_etf_s_b)) - cluster) / (
                    self.spam_num + self.gamma_etf_s_a + self.gamma_etf_s_b - cluster):
                print('e')

        # ∏𝑓∈{CS, MNR, BST, RFR}{(𝑝(𝑦𝑎,𝑟,𝑓|𝜓𝜋𝑖,f}

        # 连续变量的条件概率计算公式一（论文原公式） maybe wrong?
        author_factor_s_w = cs ** (self.psi_cs_s_a - 1) * (1 - cs) ** (self.psi_cs_s_b - 1) *\
                mnr ** (self.psi_mnr_s_a - 1) * (1 - mnr) ** (self.psi_mnr_s_b - 1) * \
                bst ** (self.psi_bst_s_a - 1) * (1 - bst) ** (self.psi_bst_s_b - 1) * \
                rfr ** (self.psi_rfr_s_a - 1) * (1 - rfr) ** (self.psi_rfr_s_b - 1)

        author_factor_n_w = cs ** (self.psi_cs_n_a - 1) * (1 - cs) ** (self.psi_cs_n_b - 1) *\
                mnr ** (self.psi_mnr_n_a - 1) * (1 - mnr) ** (self.psi_mnr_n_b - 1) * \
                bst ** (self.psi_bst_n_a - 1) * (1 - bst) ** (self.psi_bst_n_b - 1) * \
                rfr ** (self.psi_rfr_n_a - 1) * (1 - rfr) ** (self.psi_rfr_n_b - 1)

        # #连续变量的条件概率计算公式二（依旧不收敛） modify update
        # cs_factor_s = (1+cs) ** (self.psi_cs_s_a - 1) * (1 - cs) ** (self.psi_cs_s_b - 1)
        # cs_factor_n = (1+cs) ** (self.psi_cs_n_a - 1) * (1 - cs) ** (self.psi_cs_n_b - 1)
        # author_factor_cs_s = cs_factor_s / (cs_factor_s + cs_factor_n)
        # author_factor_cs_n = 1 - author_factor_cs_s
        #
        # mnr_factor_s = (1+mnr) ** (self.psi_mnr_s_a - 1) * (1 - mnr) ** (self.psi_mnr_s_b - 1)
        # mnr_factor_n = (1+mnr) ** (self.psi_mnr_n_a - 1) * (1 - mnr) ** (self.psi_mnr_n_b - 1)
        # author_factor_mnr_s = mnr_factor_s / (mnr_factor_s + mnr_factor_n)
        # author_factor_mnr_n = 1 - author_factor_mnr_s
        #
        # bst_factor_s = (1+bst) ** (self.psi_bst_s_a - 1) * (1 - bst) ** (self.psi_bst_s_b - 1)
        # bst_factor_n = (1+bst) ** (self.psi_bst_n_a - 1) * (1 - bst) ** (self.psi_bst_n_b - 1)
        # author_factor_bst_s = bst_factor_s / (bst_factor_s + bst_factor_n)
        # author_factor_bst_n = 1 - author_factor_bst_s
        #
        # rfr_factor_s = (1+rfr) ** (self.psi_rfr_s_a - 1) * (1 - rfr) ** (self.psi_rfr_s_b - 1)
        # rfr_factor_n = (1+rfr) ** (self.psi_rfr_n_a - 1) * (1 - rfr) ** (self.psi_rfr_n_b - 1)
        # author_factor_rfr_s = rfr_factor_s / (rfr_factor_s + rfr_factor_n)
        # author_factor_rfr_n = 1 - author_factor_rfr_s
        #
        # author_factor_s = author_factor_cs_s * author_factor_mnr_s * author_factor_bst_s * author_factor_rfr_s
        # author_factor_n = author_factor_cs_n * author_factor_mnr_n * author_factor_bst_n * author_factor_rfr_n

        #连续变量的条件概率计算公式三（较简易）
        # P(class|cs)
        cs_factor_s = 1/abs(cs-self.cs_s_average)
        cs_factor_n = 1/abs(cs-self.cs_n_average)
        author_factor_cs_s = cs_factor_s / (cs_factor_s + cs_factor_n)
        author_factor_cs_n = 1 - author_factor_cs_s

        # P(class|mnr)
        mnr_factor_s = 1/abs(mnr-self.mnr_s_average)
        mnr_factor_n = 1/abs(mnr-self.mnr_n_average)
        author_factor_mnr_s = mnr_factor_s / (mnr_factor_s + mnr_factor_n)
        author_factor_mnr_n = 1 - author_factor_mnr_s

        # P(class|bst)
        bst_factor_s = 1 / abs(bst - self.bst_s_average)
        bst_factor_n = 1 / abs(bst - self.bst_n_average)
        author_factor_bst_s = bst_factor_s / (bst_factor_s + bst_factor_n)
        author_factor_bst_n = 1 - author_factor_bst_s

        #P(class|rfr)
        rfr_factor_s = 1 / abs(rfr - self.rfr_s_average)
        rfr_factor_n = 1 / abs(rfr - self.rfr_n_average)
        author_factor_rfr_s = rfr_factor_s / (rfr_factor_s + rfr_factor_n)
        author_factor_rfr_n = 1 - author_factor_rfr_s

        author_factor_s = author_factor_cs_s * author_factor_mnr_s * author_factor_bst_s * author_factor_rfr_s
        author_factor_n = author_factor_cs_n * author_factor_mnr_n * author_factor_bst_n * author_factor_rfr_n


        # check two different update ways
        # if author_factor_s != author_factor_s_w and iteration >= self.burn_in:
        #     print(author_factor_s)
        #     print(author_factor_s_w)
        #     exit('different result')


        author_factor_compoments = {}
        author_factor_compoments['psi_cs_s_a'] = self.psi_cs_s_a
        author_factor_compoments['psi_cs_s_b'] = self.psi_cs_s_b
        author_factor_compoments['psi_mnr_s_a'] = self.psi_mnr_s_a
        author_factor_compoments['psi_mnr_s_b'] = self.psi_mnr_s_b
        author_factor_compoments['psi_bst_s_a'] = self.psi_bst_s_a
        author_factor_compoments['psi_bst_s_b'] = self.psi_bst_s_b
        author_factor_compoments['psi_rfr_s_a'] = self.psi_rfr_s_a
        author_factor_compoments['psi_rfr_s_b'] = self.psi_rfr_s_b

        author_factor_compoments['psi_cs_n_a'] = self.psi_cs_n_a
        author_factor_compoments['psi_cs_n_b'] = self.psi_cs_n_b
        author_factor_compoments['psi_mnr_n_a'] = self.psi_mnr_n_a
        author_factor_compoments['psi_mnr_n_b'] = self.psi_mnr_n_b
        author_factor_compoments['psi_bst_n_a'] = self.psi_bst_n_a
        author_factor_compoments['psi_bst_n_b'] = self.psi_bst_n_b
        author_factor_compoments['psi_rfr_n_a'] = self.psi_rfr_n_a
        author_factor_compoments['psi_rfr_n_b'] = self.psi_rfr_n_b

        author_factor_compoments['factor_cs_s_1'] = (1 + cs) ** (self.psi_cs_s_a - 1)
        author_factor_compoments['factor_cs_s_2'] = (1 - cs) ** (self.psi_cs_s_b - 1)
        author_factor_compoments['factor_mnr_s_1'] = (1 + mnr) ** (self.psi_mnr_s_a - 1)
        author_factor_compoments['factor_mnr_s_2'] = (1 - mnr) ** (self.psi_mnr_s_b -1)
        author_factor_compoments['factor_bst_s_1'] = (1 + bst) ** (self.psi_bst_s_a - 1)
        author_factor_compoments['factor_bst_s_2'] = (1 - bst) ** (self.psi_bst_s_b - 1)
        author_factor_compoments['factor_rfr_s_1'] = (1 + rfr) ** (self.psi_rfr_s_a - 1)
        author_factor_compoments['factor_rfr_s_2'] = (1 - rfr) ** (self.psi_rfr_s_b - 1)

        author_factor_compoments['factor_cs_n_1'] = (1 + cs) ** (self.psi_cs_n_a - 1)
        author_factor_compoments['factor_cs_n_2'] = (1 - cs) ** (self.psi_cs_n_b - 1)
        author_factor_compoments['factor_mnr_n_1'] = (1 + mnr) ** (self.psi_mnr_n_a - 1)
        author_factor_compoments['factor_mnr_n_2'] = (1 - mnr) ** (self.psi_mnr_n_b - 1)
        author_factor_compoments['factor_bst_n_1'] = (1 + bst) ** (self.psi_bst_n_a - 1)
        author_factor_compoments['factor_bst_n_2'] = (1 - bst) ** (self.psi_bst_n_b - 1)
        author_factor_compoments['factor_rfr_n_1'] = (1 + rfr) ** (self.psi_rfr_n_a - 1)
        author_factor_compoments['factor_rfr_n_2'] = (1 - rfr) ** (self.psi_rfr_n_b - 1)


        # compare strength the different factor
        # if iteration == self.iterations-2:
        #     print('----------------------')
        #     print('author_spamicity', author_spamicity_s/author_spamicity_n)
        #     print('review_factor', review_factor_s/review_factor_n)
        #     print('author_factor', author_factor_s/author_factor_n)
        #     P_spam = author_spamicity_s * review_factor_s * author_factor_s
        #     P_non_spam = author_spamicity_n * review_factor_n * author_factor_n
        #     print(P_spam/(P_non_spam+P_spam))

        # 记录每一次采样细节
        detail = {}
        detail['review'] = review
        detail['author_spamicity'] = author_spamicity_s/author_spamicity_n
        detail['review_factor'] = review_factor_s/review_factor_n
        detail['author_factor'] = author_factor_s/author_factor_n
        details.append(str(detail))
        details.append(str(author_factor_compoments))


        # calculate conditional probability using naive bayes
        P_spam = author_spamicity_s * review_factor_s * author_factor_s
        P_non_spam = author_spamicity_n * review_factor_n * author_factor_n

        # test change
        # P_spam = review_factor_s * author_factor_s
        # P_non_spam =review_factor_n * author_factor_n

        # 检查 P = 0 是如何发生的--------------------------------------------------
        if P_spam == 0:
            print('P_spam = 0')
            print(author_spamicity_s)
            print(review_factor_s)
            print(author_factor_s)
            print('self.author_spam[author]', self.author_spam[author])
            print('cluster', cluster)
            print('author_review_num', self.authors_review_num[author])
            # print('review_factor_s_compoments', review_factor_compoments)
            print('author_factor_s_compoments',author_factor_compoments)
            print(review)
            print(self.spam_num)
            exit()
        # 检查 nan 问题
        if P_spam != P_spam:
            print('P_spam 是 nan')
            if author_spamicity_s != author_spamicity_s:
                print('author_spamicity_s 是 nan')
            if review_factor_s != review_factor_s:
                print('review_factor_s 是 nan')
            if author_factor_s != author_factor_s:
                print('author_factor_s 是 nan')
                print(self.spam_num)
                print(self.non_spam_num)
                print(author_factor_compoments)
                print('rfr_s_average', self.rfr_s_average)
                print('rfr_n_average', self.rfr_n_average)
                print('rfr_s_variance', self.rfr_s_variance)
                print('rfr_n_variance', self.rfr_n_variance)
                print(review)

        if P_non_spam != P_non_spam:
            print('P_non_spam 是 nan')
            if author_spamicity_n != author_spamicity_n:
                print('author_spamicity_n 是 nan')
            if review_factor_n != review_factor_n:
                print('review_factor_n 是 nan')
            if author_factor_n != author_factor_n:
                print('author_factor_n 是 nan')
                print(cs)
                print(self.psi_cs_n_a-1)
                print(self.psi_cs_n_b-1)
                print(self.cs_n_average)
                print(self.spam_num)
                print(self.non_spam_num)
        # print('P_spam',P_spam)
        # print('P_non_spam',P_non_spam)
        # print(P_spam/P_non_spam)
        # normalization
        P_spam /= (P_spam + P_non_spam)
        # print(P_spam)
        if P_spam != P_spam:
            print('P 是 NAN')
        # -------------------------------------------------------------------------

        # 检查P > 1是如何发生的----------------------------------------------------
        if P_spam > 1:
            print(P_spam, P_non_spam)
        if P_non_spam < 0:
            print('author_spamicity_n', author_spamicity_n)
            print('review_factor_n', review_factor_n)
            print('author_factor_n', author_factor_n)
        # print('testing', P_spam)

        # sample with conditional probabilityq
        assign = np.random.binomial(1, P_spam, size=None)
        # print(assign)
        return assign

    def hyperparameter_em(self, review):
        pass

    def count_review_features(self):
        # [RA, DUP, EXT, DEV, ETF | CS, MNR, BST, RFR | type | author | ture_type], sorted by author id
        # calculate RA counts in {spam, non-spam}
        self.n_ra_s_1 = np.sum(self.spam_reviews[:, 0])
        self.n_ra_s_0 = self.spam_num - self.n_ra_s_1
        self.n_ra_n_1 = np.sum(self.non_spam_reviews[:, 0])
        self.n_ra_n_0 = self.non_spam_num - self.n_ra_n_1

        # calculate DUP counts in {spam, non-spam}
        self.n_dup_s_1 = np.sum(self.spam_reviews[:, 1])
        self.n_dup_s_0 = self.spam_num - self.n_dup_s_1
        self.n_dup_n_1 = np.sum(self.non_spam_reviews[:, 1])
        self.n_dup_n_0 = self.non_spam_num - self.n_dup_n_1

        # calculate EXT counts in {spam, non-spam}
        self.n_ext_s_1 = np.sum(self.spam_reviews[:, 2])
        self.n_ext_s_0 = self.spam_num - self.n_ext_s_1
        self.n_ext_n_1 = np.sum(self.non_spam_reviews[:, 2])
        self.n_ext_n_0 = self.non_spam_num - self.n_ext_n_1

        # calculate DEV counts in {spam, non-spam}
        self.n_dev_s_1 = np.sum(self.spam_reviews[:, 3])
        self.n_dev_s_0 = self.spam_num - self.n_dev_s_1
        self.n_dev_n_1 = np.sum(self.non_spam_reviews[:, 3])
        self.n_dev_n_0 = self.non_spam_num - self.n_dev_n_1

        # calculate ETF counts in {spam, non-spam}
        self.n_etf_s_1 = np.sum(self.spam_reviews[:, 4])
        self.n_etf_s_0 = self.spam_num - self.n_etf_s_1
        self.n_etf_n_1 = np.sum(self.non_spam_reviews[:, 4])
        self.n_etf_n_0 = self.non_spam_num - self.n_etf_n_1

    def calculate_author_features(self):
        # [RA, DUP, EXT, DEV, ETF | CS, MNR, BST, RFR | type | author | ture_type], sorted by author id

        # calculate CS average and variance
        self.cs_s_average = np.mean(self.spam_reviews[:, 5])
        self.cs_s_variance = np.var(self.spam_reviews[:, 5])
        self.cs_n_average = np.mean(self.non_spam_reviews[:, 5])
        self.cs_n_variance = np.var(self.non_spam_reviews[:, 5])

        # calculate MNR average and variance
        self.mnr_s_average = np.mean(self.spam_reviews[:, 6])
        self.mnr_s_variance = np.var(self.spam_reviews[:, 6])
        self.mnr_n_average = np.mean(self.non_spam_reviews[:, 6])
        self.mnr_n_variance = np.var(self.non_spam_reviews[:, 6])

        # calculate BST average and variance
        self.bst_s_average = np.mean(self.spam_reviews[:, 7])
        self.bst_s_variance = np.var(self.spam_reviews[:, 7])
        self.bst_n_average = np.mean(self.non_spam_reviews[:, 7])
        self.bst_n_variance = np.var(self.non_spam_reviews[:, 7])

        # calculate RFR average and variance
        self.rfr_s_average = np.mean(self.spam_reviews[:, 8])
        self.rfr_s_variance = np.var(self.spam_reviews[:, 8])
        self.rfr_n_average = np.mean(self.non_spam_reviews[:, 8])
        self.rfr_n_variance = np.var(self.non_spam_reviews[:, 8])

        # calculate (psi_a, psi_b) of CS for {spam, non-spam}
        self.psi_cs_s_a = self.cs_s_average * (
            (self.cs_s_average * (1 - self.cs_s_average) / self.cs_s_variance) - 1)
        self.psi_cs_s_b = (1 - self.cs_s_average) * (
            (self.cs_s_average * (1 - self.cs_s_average) / self.cs_s_variance) - 1)

        self.psi_cs_n_a = self.cs_n_average * (
        (self.cs_n_average * (1 - self.cs_n_average) / self.cs_n_variance) - 1)
        self.psi_cs_n_b = (1 - self.cs_n_average) * (
        (self.cs_n_average * (1 - self.cs_n_average) / self.cs_n_variance) - 1)

        # calculate (psi_a, psi_b) of MNR for {spam, non-spam}
        self.psi_mnr_s_a = self.mnr_s_average * (
            (self.mnr_s_average * (1 - self.mnr_s_average) / self.mnr_s_variance) - 1)
        self.psi_mnr_s_b = (1 - self.mnr_s_average) * (
            (self.mnr_s_average * (1 - self.mnr_s_average) / self.mnr_s_variance) - 1)

        self.psi_mnr_n_a = self.mnr_n_average * (
        (self.mnr_n_average * (1 - self.mnr_n_average) / self.mnr_n_variance) - 1)
        self.psi_mnr_n_b = (1 - self.mnr_n_average) * (
        (self.mnr_n_average * (1 - self.mnr_n_average) / self.mnr_n_variance) - 1)

        # calculate (psi_a, psi_b) of BST for {spam, non-spam}
        self.psi_bst_s_a = self.bst_s_average * (
            (self.bst_s_average * (1 - self.bst_s_average) / self.bst_s_variance) -1)
        self.psi_bst_s_b = (1 - self.bst_s_average) * (
            (self.bst_s_average * (1 - self.bst_s_average) / self.bst_s_variance) -1)

        self.psi_bst_n_a = self.bst_n_average * (
        (self.bst_n_average * (1 - self.bst_n_average) / self.bst_n_variance) - 1)
        self.psi_bst_n_b = (1 - self.bst_n_average) * (
        (self.bst_n_average * (1 - self.bst_n_average) / self.bst_n_variance) - 1)

        # calculate (psi_a, psi_b) of RFR for {spam, non-spam}
        self.psi_rfr_s_a = self.rfr_s_average * (
            (self.rfr_s_average * (1 - self.rfr_s_average) / self.rfr_s_variance) -1)
        self.psi_rfr_s_b = (1 - self.rfr_s_average) * (
            (self.rfr_s_average * (1 - self.rfr_s_average) / self.rfr_s_variance) -1)

        self.psi_rfr_n_a = self.rfr_n_average * (
        (self.rfr_n_average * (1 - self.rfr_n_average) / self.rfr_n_variance) - 1)
        self.psi_rfr_n_b = (1 - self.rfr_n_average) * (
        (self.rfr_n_average * (1 - self.rfr_n_average) / self.rfr_n_variance) - 1)


    def __update_disperse_samples(self, review, parameters='review_features'):
        if parameters not in ['review_features', 'author_features']:
            raise NameError('invalid args')
        ra, dup, ext, dev, etf, cs, mnr, bst, rfr, cluster, author, true_label = review
        # update review_features
        if parameters == 'review_features':
            # change review cluster from spam(1) to non-spam(0)
            if cluster == 1:

                self.spam_num -= 1
                self.non_spam_num += 1

                if ra == 0:
                    self.n_ra_n_0 += 1
                    self.n_ra_s_0 -= 1
                else:
                    self.n_ra_n_1 += 1
                    self.n_ra_s_1 -= 1

                if dup == 0:
                    self.n_dup_n_0 += 1
                    self.n_dup_s_0 -= 1
                else:
                    self.n_dup_n_1 += 1
                    self.n_dup_s_1 -= 1

                if ext == 0:
                    self.n_ext_n_0 += 1
                    self.n_ext_s_0 -= 1
                else:
                    self.n_ext_n_1 += 1
                    self.n_ext_s_1 -= 1

                if dev == 0:
                    self.n_dev_n_0 += 1
                    self.n_dev_s_0 -= 1
                else:
                    self.n_dev_n_1 += 1
                    self.n_dev_s_1 -= 1

                if etf == 0:
                    self.n_etf_n_0 += 1
                    self.n_etf_s_0 -= 1
                else:
                    self.n_etf_n_1 += 1
                    self.n_etf_s_1 -= 1

            # change review cluster from spam(0) to non-spam(1)
            else:

                self.spam_num += 1
                self.non_spam_num -= 1

                if ra == 0:
                    self.n_ra_s_0 += 1
                    self.n_ra_n_0 -= 1
                else:
                    self.n_ra_s_1 += 1
                    self.n_ra_n_1 -= 1

                if dup == 0:
                    self.n_dup_s_0 += 1
                    self.n_dup_n_0 -= 1
                else:
                    self.n_dup_s_1 += 1
                    self.n_dup_n_1 -= 1

                if ext == 0:
                    self.n_ext_s_0 += 1
                    self.n_ext_n_0 -= 1
                else:
                    self.n_ext_s_1 += 1
                    self.n_ext_n_1 -= 1

                if dev == 0:
                    self.n_dev_s_0 += 1
                    self.n_dev_n_0 -= 1
                else:
                    self.n_dev_s_1 += 1
                    self.n_dev_n_1 -= 1

                if etf == 0:
                    self.n_etf_s_0 += 1
                    self.n_etf_n_0 -= 1
                else:
                    self.n_etf_s_1 += 1
                    self.n_etf_n_1 -= 1

    # update author_features
    def __update_continuous_samples(self):

        # update psi_f_k
        self.calculate_author_features()
        # normalization
        self.normalize_psi_k_f()


    def normalize_psi_k_f(self):

        # normalize psi_f_k by the smaller one of all features in spam and non-spam
        normalization_factor = min(self.psi_cs_s_a+self.psi_cs_s_b, self.psi_cs_n_a+self.psi_cs_n_b,
                                   self.psi_mnr_s_a+self.psi_mnr_s_b, self.psi_mnr_n_a+self.psi_mnr_n_b,
                                   self.psi_bst_s_a+self.psi_bst_s_b, self.psi_bst_n_a+self.psi_bst_n_a,
                                   self.psi_rfr_s_a+self.psi_rfr_s_b, self.psi_rfr_n_a+self.psi_rfr_n_b)


        # normalize psi_cs_k
        if (self.psi_cs_s_a + self.psi_cs_s_b) > (self.psi_cs_n_a + self.psi_cs_n_b):

            self.psi_cs_s_a = self.psi_cs_s_a / (self.psi_cs_s_a + self.psi_cs_s_b) * \
                              (self.psi_cs_n_a + self.psi_cs_n_b)
            self.psi_cs_s_b = (self.psi_cs_n_a + self.psi_cs_n_b) - self.psi_cs_s_a

        else:

            self.psi_cs_n_a = self.psi_cs_n_a / (self.psi_cs_n_a + self.psi_cs_n_b) * \
                              (self.psi_cs_s_a + self.psi_cs_s_b)
            self.psi_cs_n_b = (self.psi_cs_s_a + self.psi_cs_s_b) - self.psi_cs_n_a

        # self.psi_cs_s_a = self.psi_cs_s_a / (self.psi_cs_s_a+self.psi_cs_s_b) * normalization_factor
        # self.psi_cs_s_b = self.psi_cs_s_b / (self.psi_cs_s_a+self.psi_cs_s_b) * normalization_factor
        #
        # self.psi_cs_n_a = self.psi_cs_n_a / (self.psi_cs_n_a+self.psi_cs_n_b) * normalization_factor
        # self.psi_cs_n_b = self.psi_cs_n_b / (self.psi_cs_n_a+self.psi_cs_n_b) * normalization_factor

        # normalize psi_mnr_k
        if (self.psi_mnr_s_a + self.psi_mnr_s_b) > (self.psi_mnr_n_a + self.psi_mnr_n_b):

            self.psi_mnr_s_a = self.psi_mnr_s_a / (self.psi_mnr_s_a + self.psi_mnr_s_b) * \
                              (self.psi_mnr_n_a + self.psi_mnr_n_b)
            self.psi_mnr_s_b = (self.psi_mnr_n_a + self.psi_mnr_n_b) - self.psi_mnr_s_a

        else:
            self.psi_mnr_n_a = self.psi_mnr_n_a / (self.psi_mnr_n_a + self.psi_mnr_n_b) * \
                              (self.psi_mnr_s_a + self.psi_mnr_s_b)
            self.psi_mnr_n_b = (self.psi_mnr_s_a + self.psi_mnr_s_b) - self.psi_mnr_n_a

        # self.psi_mnr_s_a = self.psi_mnr_s_a / (self.psi_mnr_s_a + self.psi_mnr_s_b) * normalization_factor
        # self.psi_mnr_s_b = self.psi_mnr_s_b / (self.psi_mnr_s_a + self.psi_mnr_s_b) * normalization_factor
        #
        # self.psi_mnr_n_a = self.psi_mnr_n_a / (self.psi_mnr_n_a + self.psi_mnr_n_b) * normalization_factor
        # self.psi_mnr_n_b = self.psi_mnr_n_b / (self.psi_mnr_n_a + self.psi_mnr_n_b) * normalization_factor

        # normalize psi_bst_k
        if (self.psi_bst_s_a + self.psi_bst_s_b) > (self.psi_bst_n_a + self.psi_bst_n_b):

            self.psi_bst_s_a = self.psi_bst_s_a / (self.psi_bst_s_a + self.psi_bst_s_b) * \
                              (self.psi_bst_n_a + self.psi_bst_n_b)
            self.psi_bst_s_b = (self.psi_bst_n_a + self.psi_bst_n_b) - self.psi_bst_s_a

        else:

            self.psi_bst_n_a = self.psi_bst_n_a / (self.psi_bst_n_a + self.psi_bst_n_b) * \
                              (self.psi_bst_s_a + self.psi_bst_s_b)
            self.psi_bst_n_b = (self.psi_bst_s_a + self.psi_bst_s_b) - self.psi_bst_n_a

        # self.psi_bst_s_a = self.psi_bst_s_a / (self.psi_bst_s_a + self.psi_bst_s_b) * normalization_factor
        # self.psi_bst_s_b = self.psi_bst_s_b / (self.psi_bst_s_a + self.psi_bst_s_b) * normalization_factor
        #
        # self.psi_bst_n_a = self.psi_bst_n_a / (self.psi_bst_n_a + self.psi_bst_n_b) * normalization_factor
        # self.psi_bst_n_b = self.psi_bst_n_b / (self.psi_bst_n_a + self.psi_bst_n_b) * normalization_factor

        # normalize psi_rfr_k
        if (self.psi_rfr_s_a + self.psi_rfr_s_b) > (self.psi_rfr_n_a + self.psi_rfr_n_b):

            self.psi_rfr_s_a = self.psi_rfr_s_a / (self.psi_rfr_s_a + self.psi_rfr_s_b) * \
                              (self.psi_rfr_n_a + self.psi_rfr_n_b)
            self.psi_rfr_s_b = (self.psi_rfr_n_a + self.psi_rfr_n_b) - self.psi_rfr_s_a

        else:

            self.psi_rfr_n_a = self.psi_rfr_n_a / (self.psi_rfr_n_a + self.psi_rfr_n_b) * \
                              (self.psi_rfr_s_a + self.psi_rfr_s_b)
            self.psi_rfr_n_b = (self.psi_rfr_s_a + self.psi_rfr_s_b) - self.psi_rfr_n_a

        # self.psi_rfr_s_a = self.psi_rfr_s_a / (self.psi_rfr_s_a + self.psi_rfr_s_b) * normalization_factor
        # self.psi_rfr_s_b = self.psi_rfr_s_b / (self.psi_rfr_s_a + self.psi_rfr_s_b) * normalization_factor
        #
        # self.psi_rfr_n_a = self.psi_rfr_n_a / (self.psi_rfr_n_a + self.psi_rfr_n_b) * normalization_factor
        # self.psi_rfr_n_b = self.psi_rfr_n_b / (self.psi_rfr_n_a + self.psi_rfr_n_b) * normalization_factor


    def __collect_parameters(self):
        # collect θ and ψ
        spam_parameters = {'θ_ra': (self.gamma_ra_s_a + self.n_ra_s_1, self.gamma_ra_s_b + self.n_ra_s_0),
                           'θ_dup': (self.gamma_dup_s_a + self.n_dup_s_1, self.gamma_dup_s_b + self.n_dup_s_0),
                           'θ_ext': (self.gamma_ext_s_a + self.n_ext_s_1, self.gamma_ext_s_b + self.n_ext_s_0),
                           'θ_dev': (self.gamma_dev_s_a + self.n_dev_s_1, self.gamma_dev_s_b + self.n_dev_s_0),
                           'θ_etf': (self.gamma_etf_s_a + self.n_etf_s_1, self.gamma_etf_s_b + self.n_etf_s_0), # review features
                           'ψ_cs': (self.psi_cs_s_a, self.psi_cs_s_b),
                           'ψ_mnr': (self.psi_mnr_s_a, self.psi_mnr_s_b),
                           'ψ_bst': (self.psi_bst_s_a, self.psi_bst_s_b),
                           'ψ_rfr': (self.psi_rfr_s_a, self.psi_rfr_s_b)} #author features

        non_spam_parameters = {'θ_ra': (self.gamma_ra_n_a + self.n_ra_n_1, self.gamma_ra_n_b + self.n_ra_n_0),
                               'θ_dup': (self.gamma_dup_n_a + self.n_dup_n_1, self.gamma_dup_n_b + self.n_dup_n_0),
                               'θ_ext': (self.gamma_ext_n_a + self.n_ext_n_1, self.gamma_ext_n_b + self.n_ext_n_0),
                               'θ_dev': (self.gamma_dev_n_a + self.n_dev_n_1, self.gamma_dev_n_b + self.n_dev_n_0),
                               'θ_etf': (self.gamma_etf_n_a + self.n_etf_n_1, self.gamma_etf_n_b + self.n_etf_n_0), # review features
                               'ψ_cs': (self.psi_cs_n_a, self.psi_cs_n_b),
                               'ψ_mnr': (self.psi_mnr_n_a, self.psi_mnr_n_b),
                               'ψ_bst': (self.psi_bst_n_a, self.psi_bst_n_b),
                               'ψ_rfr': (self.psi_rfr_n_a, self.psi_rfr_n_b)}  # author features

        return spam_parameters, non_spam_parameters

    def collect_hyperparameters(self):
        # collect hyper parameters gamma γ and alpha α
        pass

    def show_parameters(self, para='non-spam', vision=True):
        # [RA, DUP, EXT, DEV, ETF | CS, MNR, BST, RFR | type | author], sorted by author id
        # x 轴
        # print(self.n_ra_n_1)
        x = np.arange(0.01, 1, 0.01)
        # parameters dict
        sp, nsp = self.__collect_parameters()
        figures = plt.figure()
        print('--------'+para+' parameters'+'-------')
        if para == 'non-spam':
            for i, key in enumerate(nsp):
                print(str(key)+' average value is: '+str(nsp[key][0]/(nsp[key][0]+nsp[key][1])))
                print(nsp[key][0], nsp[key][1])
                figures.add_subplot(3, 3, i+1)
                plt.plot(x, scipy.stats.beta.pdf(x, nsp[key][0], nsp[key][1]))
                plt.title(str(key))
                # plt.title(str(key)+':a='+str(nsp[key][0])+',b='+str(nsp[key][1]))
            plt.show()
        else:
            for i, key in enumerate(sp):
                print(str(key)+' average value is: '+str(sp[key][0]/(sp[key][0]+sp[key][1])))
                print(sp[key][0], sp[key][1])
                figures.add_subplot(3, 3, i+1)
                plt.plot(x, scipy.stats.beta.pdf(x, sp[key][0], sp[key][1]))
                plt.title(str(key))
                # plt.title(str(key)+':a='+str(sp[key][0])+',b='+str(sp[key][1]))
            plt.show()

    def train(self, data):
        pass

    def test(self):
        pass

    def preditc(self, data, show_prob=False):
        P_spam = ''
        P_non_spam = ''

        if int(P_spam) > 0.5:
            return True
        else:
            return False

    def Precision(self):
        predict = self.data[:, -3]
        true_label = self.data[:, -1]

        precision = np.sum(predict == true_label)/self.data_num
        self.precision = max(precision, 1-precision)

        # print('随机分配准确率', self.raw_precision)
        # print('ASM准确率：', precision)

        return self.precision


