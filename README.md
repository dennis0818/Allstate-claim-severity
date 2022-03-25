# Allstate-claim-severity

## 一、项目简介
当你被一场严重的车祸摧毁时，你会把注意力放在最重要的事情上:家人、朋友和其他爱的人。你的保险代理是你最不愿意花时间和精力的地方。这就是为什么好事达(Allstate)，一家美国的个人保险公司，不断寻求新的思路，为他们所保护的超过1600万户家庭改善他们的理赔服务。好事达目前正在开发自动化的方法来预测索赔的成本，从而预测索赔的严重程度。 

## 二、数据描述
来自kaggle赛题数据集
此数据集中的每一行表示一个保险索赔，预测“损失”列的值。以“cat”开头的变量是分类变量，而以“cont”开头的变量是连续变量。

训练集：train.csv
测试集：test.csv

## 三、预测方法
清洗数据空值
用XGBoost算法，调整参数组合进行预测
