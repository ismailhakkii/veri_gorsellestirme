import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini okumak için
df = pd.read_csv('telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# İlk birkaç satır yazdırmak için
print(df.info())
print(df.head())

# Eksik veri kontrolü
print(df.isnull().sum())


# Sayısal sütunlar için istatistikler
print(df.describe())

# Aylık hizmet süresi dağılımı (tenure)
plt.figure(figsize=(10, 5))
sns.histplot(df['tenure'], kde=True, color='blue')
plt.title('Müşterilerin Aylık Hizmet Süresi Dağılımı')
plt.xlabel('Aylık Hizmet Süresi')
plt.ylabel('Müşteri Sayısı')
plt.show()

# Aylık ücret dağılımı
plt.figure(figsize=(10, 5))
sns.histplot(df['MonthlyCharges'], kde=True, color='green')
plt.title('Aylık Ücret Dağılımı')
plt.xlabel('Aylık Ücret')
plt.ylabel('Müşteri Sayısı')
plt.show()

# Toplam ödeme dağılımı
plt.figure(figsize=(10, 5))
sns.histplot(df['TotalCharges'], kde=True, color='red')
plt.title('Toplam Ödeme Dağılımı')
plt.xlabel('Toplam Ödeme')
plt.ylabel('Müşteri Sayısı')
plt.show()

# Kategorik değişkenlerin dağılımını ve churn ilişkisini incelemek için
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df[col], hue=df['Churn'], palette='Set2')
    plt.title(f'{col} Sütununun Dağılımı ve Churn İlişkisi')
    plt.xlabel(col)
    plt.ylabel('Müşteri Sayısı')
    plt.show()

# Korelasyon matrisi oluşturmak için
corr_matrix = df.corr()

# Korelasyon ısı haritası çizmek için
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasyon Isı Haritası')
plt.show()
