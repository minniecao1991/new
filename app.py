# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import squarify
import plotly.express as px
from sklearn.preprocessing import RobustScaler 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.cluster.hierarchy as sch 
import pickle
import joblib
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import io

st.title("Customers Segmentation")

products_df = pd.read_csv('Products_with_Categories.csv')
transactions_df = pd.read_csv('Transactions.csv')
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], format='%d-%m-%Y')
transactions_df['order_id'] = transactions_df.groupby(['Member_number', 'Date']).ngroup() + 1
merged_df = pd.merge(transactions_df, products_df, on='productId', how='left')
merged_df['Total_Cost'] = merged_df['price'] * merged_df['items']
merged_df['Month'] = merged_df['Date'].dt.to_period('M')
monthly_category_transactions = merged_df.groupby(['Month', 'Category']).size().reset_index(name='Transaction_Count')
pivot_table = monthly_category_transactions.pivot(index='Category', columns='Month', values='Transaction_Count').fillna(0)
category_counts = products_df.groupby('Category')['productName'].nunique().reset_index(name='Product_Count')
category_counts = category_counts.sort_values(by='Product_Count', ascending=False)
top_products = merged_df.groupby('productName')['Total_Cost'].sum().reset_index()
top_10_products = top_products.sort_values(by='Total_Cost', ascending=False).head(10)
top_products = merged_df.groupby('productName')['items'].sum().sort_values(ascending=False).head(10)
top_categories = merged_df.groupby('Category')['Total_Cost'].sum().reset_index()
top_10_categories = top_categories.sort_values(by='Total_Cost', ascending=False).head(10)
fresh_food_df = merged_df[merged_df['Category'] == 'Fresh Food']
top_fresh_food = fresh_food_df.groupby('productName')['Total_Cost'].sum().reset_index()
top_10_fresh_food = top_fresh_food.sort_values(by='Total_Cost', ascending=False).head(10)

df = merged_df
current_date = df['Date'].max()
rfm_df = df.groupby('Member_number').agg({
'Date': lambda x: (current_date - x.max()).days,  # Recency: Số ngày từ giao dịch cuối cùng
'order_id': 'nunique',                            # Frequency: Số đơn hàng duy nhất
'Total_Cost': 'sum'                               # Monetary: Tổng chi tiêu
}).reset_index()
rfm_df.columns = ['Member_number', 'Recency', 'Frequency', 'Monetary']
rfm_df = rfm_df.sort_values('Monetary', ascending=False)
# Create labels for Recency, Frequency, Monetary
r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
f_labels = range(1, 5)
m_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(rfm_df['Recency'].rank(method='first'), q=4, labels=r_labels)
f_groups = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_groups = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=4, labels=m_labels)
rfm_df = rfm_df.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
rfm_df['RFM_Segment'] = rfm_df.apply(join_rfm, axis=1)
rfm_count_unique = rfm_df.groupby('RFM_Segment')['RFM_Segment'].nunique()
rfm_df['RFM_Score'] = rfm_df[['R','F','M']].sum(axis=1)
def rfm_level(df):
    if df['RFM_Score'] == 12:
        return 'Best Customers'  # High recency, frequency, and monetary
    elif df['R'] == 1 and df['F'] == 1 and df['M'] == 1:
        return 'New Customers'  # Very low recency, frequency, and monetary
    elif df['M'] == 4:
        return 'Big Spenders'  # High monetary
    elif df['F'] == 4:
        return 'Loyal Customers'  # High frequency
    elif df['R'] == 4:
        return 'Active Customers'  # High recency
    else:
        return 'At-Risk/Occasional'  # All other cases
rfm_df['RFM_Level'] = rfm_df.apply(rfm_level, axis=1)
output_file = 'rfm_df.csv'
rfm_df.to_csv(output_file, index=True, encoding='utf-8')

menu = ["Overview", "Build Project", "Manual RFM","Kmeans_RFM","Kmeans_RFM_bigdata","Hireachical_clustering","Tra cứu nhóm khách hàng"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Overview':    
    st.subheader("Giới thiệu project")
    st.write("""
    Dự án này được thiết kế nhằm hỗ trợ **chủ cửa hàng X** quản lý và phân tích dữ liệu khách hàng một cách hiệu quả, từ đó tối ưu hóa chiến lược kinh doanh. Dưới đây là những nôi dung của dự án:

    1. **Giới thiệu dự án**:  
       Ứng dụng được xây dựng dành riêng cho **chủ cửa hàng X**, giúp phân tích hành vi khách hàng dựa trên dữ liệu giao dịch và tương tác. Mục tiêu là cung cấp một công cụ trực quan, dễ sử dụng để hỗ trợ việc ra quyết định kinh doanh.

    2. **Kết quả đạt được**:  
       Dự án đã thành công trong việc **xác định các phân nhóm khách hàng** dựa trên các đặc điểm như thói quen mua sắm, sở thích, và mức độ chi tiêu. Các phân nhóm này giúp chủ cửa hàng hiểu rõ hơn về đối tượng khách hàng và xây dựng chiến lược tiếp cận phù hợp.

    3. **Lợi ích cho người dùng**:  
       Với giao diện thân thiện và các công cụ phân tích tích hợp, người dùng có thể **dễ dàng xác định khách hàng tiềm năng**, từ đó cá nhân hóa các chiến dịch tiếp thị và nâng cao hiệu quả kinh doanh.
    
    """)  

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("#### Data Preprocessing")
    st.write("##### Show data:")
    st.table(products_df.head(5)) 

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))  # Tạo figure và axes với kích thước 10x6
    sns.histplot(data=products_df, x='price', bins=20, kde=True, ax=ax)  # Vẽ histogram với KDE
    # Đặt tiêu đề và nhãn
    ax.set_title('Phân bố giá sản phẩm')
    ax.set_xlabel('Giá (Price)')
    ax.set_ylabel('Số lượng')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig1, ax1 = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=category_counts, x='Category', y='Product_Count', palette='viridis', ax=ax1)
    # Tùy chỉnh biểu đồ
    ax1.set_title('Số lượng sản phẩm theo danh mục', fontsize=14)
    ax1.set_xlabel('Danh mục (Category)', fontsize=12)
    ax1.set_ylabel('Số lượng sản phẩm (Product_Count)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelright=False, labelleft=True)  # Xoay nhãn trục x 45 độ
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm số liệu trên mỗi cột
    for i, v in enumerate(category_counts['Product_Count']):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig1)

    st.table(merged_df.head(5))

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(14, 8))  # Tạo figure và axes với kích thước 14x8
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Số lượng giao dịch'}, ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Số lượng giao dịch theo thời gian và danh mục (Heatmap)')
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Danh mục')
    ax.tick_params(axis='x', rotation=45)  # Xoay nhãn trục x 45 độ
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    st.table(merged_df.describe())

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))  # Tạo figure và axes với kích thước 10x6
    sns.histplot(data=merged_df, x='Total_Cost', bins=20, kde=True, ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Phân bố tổng chi phí mỗi giao dịch')
    ax.set_xlabel('Tổng chi phí (Total Cost)')
    ax.set_ylabel('Số lượng')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=top_10_products, x='productName', y='Total_Cost', palette='viridis', ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 mặt hàng có giá trị Total_Cost cao nhất', fontsize=14)
    ax.set_xlabel('Tên mặt hàng', fontsize=12)
    ax.set_ylabel('Doanh thu (Total_Cost)', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelright=False, labelleft=True)  # Xoay nhãn trục x 45 độ
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm chú thích số trên mỗi cột
    for i, v in enumerate(top_10_products['Total_Cost']):
        ax.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    top_products.plot(kind='bar', ax=ax)  # Vẽ biểu đồ cột từ top_products
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 sản phẩm được mua nhiều nhất')
    ax.set_xlabel('Tên sản phẩm')
    ax.set_ylabel('Tổng số lượng mua')
    ax.tick_params(axis='x', rotation=45)  # Xoay nhãn trục x 45 độ
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.countplot(y='Category', data=merged_df, order=merged_df['Category'].value_counts().index, ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Số lượng giao dịch theo danh mục')
    ax.set_xlabel('Số lượng giao dịch')
    ax.set_ylabel('Danh mục')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=top_10_categories, y='Category', x='Total_Cost', palette='viridis', ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 danh mục có doanh thu cao nhất', fontsize=14)
    ax.set_xlabel('Tổng doanh thu (Total_Cost)', fontsize=12)
    ax.set_ylabel('Danh mục (Category)', fontsize=12)
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm chú thích số trên mỗi cột
    for i, v in enumerate(top_10_categories['Total_Cost']):
        ax.text(v + 0.5, i, f'{v:.2f}', va='center')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=top_10_fresh_food, y='productName', x='Total_Cost', palette='viridis', ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 mặt hàng trong danh mục Fresh Food có doanh thu cao nhất', fontsize=14)
    ax.set_xlabel('Tổng doanh thu (Total_Cost)', fontsize=12)
    ax.set_ylabel('Tên mặt hàng (productName)', fontsize=12)
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm chú thích số trên mỗi cột
    for i, v in enumerate(top_10_fresh_food['Total_Cost']):
        ax.text(v + 0.5, i, f'{v:.2f}', va='center')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

elif choice == 'Manual RFM':
    st.subheader("Manual RFM")
    
    # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg = rfm_df.groupby('RFM_Level').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)
    # Reset the index
    rfm_agg = rfm_agg.reset_index()
    
    st.table(rfm_df.head())

    # Tạo figure với kích thước tổng thể
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))  # 3 hàng, 1 cột, kích thước 12x6
    # Vẽ phân phối của 'Recency'
    axes[0].hist(rfm_df['Recency'], bins=20, edgecolor='black')  # Histogram với 20 bins
    axes[0].set_title('Distribution of Recency')
    axes[0].set_xlabel('Recency')
    # Vẽ phân phối của 'Frequency'
    axes[1].hist(rfm_df['Frequency'], bins=10, edgecolor='black')  # Histogram với 10 bins
    axes[1].set_title('Distribution of Frequency')
    axes[1].set_xlabel('Frequency')
    # Vẽ phân phối của 'Monetary'
    axes[2].hist(rfm_df['Monetary'], bins=20, edgecolor='black')  # Histogram với 20 bins
    axes[2].set_title('Distribution of Monetary')
    axes[2].set_xlabel('Monetary')
    # Đảm bảo bố cục gọn gàng
    plt.tight_layout()
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    st.table(rfm_df['RFM_Level'].value_counts())

    st.table(rfm_agg)

    # Định nghĩa từ điển màu sắc
    colors_dict = {
        'Active Customers': 'yellow',
        'Big Spenders': 'royalblue',
        'Occasional Customers': 'cyan',
        'Lost Customers': 'red',
        'Loyal Customers': 'purple',
        'New Customers': 'green',
        'Best Customers': 'gold'
    }
    # Tạo figure và axes
    fig = plt.figure()  # Tạo figure
    ax = fig.add_subplot()  # Thêm subplot
    fig.set_size_inches(14, 10)  # Đặt kích thước 14x10
    # Vẽ treemap
    squarify.plot(
        sizes=rfm_agg['Count'],  # Kích thước ô dựa trên số lượng khách hàng
        text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': 'sans serif'},  # Tùy chỉnh văn bản
        color=colors_dict.values(),  # Gán màu từ từ điển
        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                      for i in range(0, len(rfm_agg))],  # Nhãn với thông tin chi tiết
        alpha=0.5  # Độ trong suốt
    )
    # Tùy chỉnh biểu đồ
    plt.title("Customers Segments", fontsize=26, fontweight="bold")
    plt.axis('off')  # Tắt trục
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ phân tán
    fig = px.scatter(
        rfm_agg,
        x="RecencyMean",
        y="MonetaryMean",
        size="FrequencyMean",
        color="RFM_Level",
        hover_name="RFM_Level",
        size_max=100
    )
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif choice == 'Kmeans_RFM':
    from sklearn.cluster import KMeans
    rfm_df= pd.read_csv('rfm_df.csv')
    df_now = rfm_df[['Recency','Frequency','Monetary']]
    rfm_df['Log_Recency'] = np.log1p(rfm_df['Recency'])
    rfm_df['Log_Frequency'] = np.log1p(rfm_df['Frequency'])
    rfm_df['Log_Monetary'] = np.log1p(rfm_df['Monetary'])
    scaler = RobustScaler()
    rfm_df[['Scaled_Log_Recency', 'Scaled_Log_Frequency', 'Scaled_Log_Monetary']] = scaler.fit_transform(
        rfm_df[['Log_Recency', 'Log_Frequency', 'Log_Monetary']])
    # Elbow Method để chọn k
    X = rfm_df[['Scaled_Log_Recency', 'Scaled_Log_Frequency', 'Scaled_Log_Monetary']]
    # Danh sách chứa SSE cho mỗi giá trị k
    sse = {}
    # Chạy KMeans cho các giá trị k từ 1 đến 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse[k] = kmeans.inertia_  
    # Lấy các cột đã chuẩn hóa từ rfm_df
    df_now_scaled = rfm_df[['Scaled_Log_Recency', 'Scaled_Log_Frequency', 'Scaled_Log_Monetary']]
    # Thực hiện phân cụm với k = 4
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(df_now_scaled)
    # Gán nhãn phân cụm vào cột 'Cluster' trong rfm_df
    rfm_df['Cluster'] = kmeans.labels_
    # Tính trung bình và đếm số lượng cho từng cụm
    rfm_agg2 = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)
    # Đổi tên cột
    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    # Tính phần trăm
    rfm_agg2['Percent'] = round((rfm_agg2['Count'] / rfm_agg2.Count.sum()) * 100, 2)
    rfm_agg2 = rfm_agg2.reset_index()

    # Đổi kiểu dữ liệu cột Cluster
    rfm_agg2['Cluster'] = 'Cluster ' + rfm_agg2['Cluster'].astype('str')

    st.table(rfm_df.head())

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))  # Tạo figure và axes với kích thước 10x6
    ax.plot(list(sse.keys()), list(sse.values()), marker='o')  # Vẽ đường với điểm đánh dấu
    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Số cụm (k)')
    ax.set_ylabel('SSE (Sum of Squared Errors)')
    ax.set_title('Elbow Method để chọn k')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ phân tán
    fig = px.scatter(
        rfm_agg2,
        x="RecencyMean",
        y="MonetaryMean",
        size="FrequencyMean",
        color="Cluster",
        hover_name="Cluster",
        size_max=100
    )
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Định nghĩa từ điển màu sắc
    colors_dict2 = {
        'Cluster0': 'yellow',
        'Cluster1': 'royalblue',
        'Cluster2': 'cyan',
        'Cluster3': 'red',
        'Cluster4': 'purple',
        'Cluster5': 'green',
        'Cluster6': 'gold'
    }
    # Tạo figure và axes
    fig = plt.figure()  # Tạo figure
    ax = fig.add_subplot()  # Thêm subplot
    fig.set_size_inches(14, 10)  # Đặt kích thước 14x10
    # Vẽ treemap
    squarify.plot(sizes=rfm_agg2['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict2.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                        for i in range(0, len(rfm_agg2))], alpha=0.5 )
    # Tùy chỉnh biểu đồ
    plt.title("Customers Segments", fontsize=26, fontweight="bold")
    plt.axis('off')  # Tắt trục
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

elif choice == "Kmeans_RFM_bigdata":
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans
    rfm_df= pd.read_csv('rfm_df.csv')
    spark = SparkSession.builder.appName("RFM Analysis with K-Means").getOrCreate()
    rfm_df = spark.read.csv("rfm_df.csv", header=True, inferSchema=True)
    df_now = rfm_df.select("Recency", "Frequency", "Monetary")
    # Chuyển đổi dữ liệu thành vector
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    df_vector = assembler.transform(df_now)
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_vector)
    df_scaled = scaler_model.transform(df_vector)
    # Tính SSE cho Elbow Method
    sse = {}
    for k in range(2, 20):  # Bắt đầu từ 2 vì k=1 không hợp lệ trong PySpark
        kmeans = KMeans(featuresCol="scaled_features", k=k, seed=42)
        model = kmeans.fit(df_scaled)
        sse[k] = model.summary.trainingCost
    
    # Chuẩn bị dữ liệu
    k_values = list(sse.keys())
    sse_values = list(sse.values())
    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(14, 10))  # Tạo figure và axes với kích thước 14x10
    ax.plot(k_values, sse_values, "bx-")  # Vẽ đường với định dạng "bx-" (blue, x markers)
    # Tùy chỉnh biểu đồ
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("SSE")
    ax.set_title("Elbow Method for Optimal k")
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Phân cụm với k=4 (giả sử từ Elbow)
    kmeans = KMeans(featuresCol="scaled_features", k=4, seed=42)
    model = kmeans.fit(df_scaled)
    df_clustered = model.transform(df_scaled).select("Recency", "Frequency", "Monetary", col("prediction").alias("Cluster"))

    # Tổng hợp dữ liệu theo cụm
    rfm_agg = df_clustered.groupBy("Cluster").agg({
        "Recency": "avg",
        "Frequency": "avg",
        "Monetary": "avg",
        "Cluster": "count"
    }).withColumnRenamed("count(Cluster)", "Count")

    # Chuyển sang pandas DataFrame và vẽ treemap
    rfm_agg_pd = rfm_agg.toPandas()
    colors_dict = {0: "yellow", 1: "royalblue", 2: "cyan", 3: "red"}
    # Tạo figure
    fig, ax = plt.subplots(figsize=(14, 10))  # Tạo figure và axes với kích thước 14x10
    # Vẽ treemap
    squarify.plot(
        sizes=rfm_agg_pd["Count"],  # Kích thước ô dựa trên số lượng khách hàng
        color=[colors_dict[i] for i in rfm_agg_pd["Cluster"]],  # Gán màu từ từ điển dựa trên Cluster
        label=[
            f"Cluster {row['Cluster']}\n{row['avg(Recency)']:.0f} days\n{row['avg(Frequency)']:.0f} orders\n{row['avg(Monetary)']:.0f} $\n{row['Count']:.0f} customers ({row['Count']/rfm_agg_pd['Count'].sum()*100:.0f}%)"
            for _, row in rfm_agg_pd.iterrows()
        ],  # Nhãn với thông tin chi tiết
        alpha=0.5,  # Độ trong suốt
        text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"}  # Tùy chỉnh văn bản
    )
    # Tùy chỉnh biểu đồ
    plt.title("Customer Segments", fontsize=26, fontweight="bold")
    plt.axis("off")  # Tắt trục
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

elif choice == "Hireachical_clustering":
    from sklearn.preprocessing import StandardScaler
    rfm_df= pd.read_csv('rfm_df.csv')
    rfm_data = rfm_df[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    dendrogram = sch.dendrogram(sch.linkage(rfm_scaled, method='ward'), ax=ax)  # Vẽ dendrogram
    # Tùy chỉnh biểu đồ
    ax.set_title('Dendrogram for Hierarchical Clustering')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Euclidean Distance')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Chọn số cụm hợp lý dựa trên dendrogram (thường là 4-6 cụm)
    num_clusters = 4  
    # Phân cụm sử dụng Hierarchical Clustering
    clusters = fcluster(sch.linkage(rfm_scaled, method='ward'), num_clusters, criterion='maxclust')
    # Gán nhãn cụm vào DataFrame
    rfm_df['Cluster'] = clusters
    # Hiển thị số lượng khách hàng trong mỗi cụm
    st.table(rfm_df['Cluster'].value_counts())  
    # Tính giá trị trung bình của mỗi cụm
    rfm_cluster_analysis = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    st.table(rfm_cluster_analysis)

    # Tính tổng số khách hàng trong mỗi cụm
    cluster_sizes = rfm_df['Cluster'].value_counts().sort_index()
    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 6))  # Tạo figure và axes với kích thước 10x6
    # Vẽ Treemap
    squarify.plot(
        sizes=cluster_sizes,  # Kích thước ô dựa trên số lượng khách hàng trong mỗi cụm
        label=[f'Cluster {i}' for i in cluster_sizes.index],  # Nhãn cho từng cụm
        alpha=0.7,  # Độ trong suốt
        color=['red', 'blue', 'green', 'orange']  # Danh sách màu cố định
    )
    # Tùy chỉnh biểu đồ
    plt.title("Phân bố khách hàng theo cụm")
    plt.axis("off")  # Ẩn trục
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

elif choice=='Tra cứu nhóm khách hàng':
    pipeline = joblib.load('customer_segmentation_pipeline.pkl')
    cluster_to_group = {
        0: 'Loyal Customers', #Mua gần đây, tần suất ổn định, chi tiêu khá cao.
        1: 'At-Risk Customers', #Lâu không mua, tần suất thấp, chi tiêu không nổi bật.
        2: 'VIP', #Mua thường xuyên, chi tiêu cao, dù không phải gần đây nhất.
        3: 'Lost Customers' #Lâu không mua, hiếm khi mua, chi tiêu rất thấp.
    }
    # Chọn nhập mã khách hàng hoặc nhập thông tin khách hàng vào dataframe
    st.write("##### 1. Chọn cách nhập thông tin khách hàng")
    type = st.radio("Chọn cách nhập thông tin khách hàng", options=["Nhập mã khách hàng", "Nhập thông tin khách hàng vào dataframe","Tải file Excel/CSV"])
    if type == "Nhập mã khách hàng":
        # Nếu người dùng chọn nhập mã khách hàng
        st.subheader("Nhập mã khách hàng")
        # Tạo điều khiển để người dùng nhập mã khách hàng
        customer_id = st.text_input("Nhập mã khách hàng")
        # Nếu người dùng nhập mã khách hàng, thực hiện các xử lý tiếp theo
        # Đề xuất khách hàng thuộc cụm nào
        # In kết quả ra màn hình
        st.write("Mã khách hàng:", customer_id)
        if customer_id:  # Kiểm tra nếu có mã khách hàng
            try:
                customer_data = rfm_df[rfm_df['Member_number'] == int(customer_id)][['Recency', 'Frequency', 'Monetary','RFM_Level']]
                if not customer_data.empty:
                    st.write("Thông tin RFM:", customer_data)
                    cluster = pipeline.predict(customer_data[['Recency', 'Frequency', 'Monetary']])
                    group_name = cluster_to_group[cluster[0]]
                    st.write(f"Khách hàng thuộc cụm: {group_name}")
                    rfm_level = customer_data['RFM_Level'].iloc[0]
                    st.write(f"Khách hàng thuộc cụm theo tập luận RFM: {rfm_level}")
                else:
                    st.write("Không tìm thấy khách hàng với mã này.")
            except ValueError:
                st.write("Vui lòng nhập mã khách hàng hợp lệ (số nguyên).")
    elif type == "Nhập thông tin khách hàng vào dataframe":
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.write("##### 2. Thông tin khách hàng")
        # Tạo điều khiển table để người dùng nhập thông tin khách hàng trực tiếp trên table
        st.write("Nhập thông tin khách hàng")
        # Tạo dataframe để người dùng nhập thông tin khách hàng
        # Tạo danh sách tạm để lưu thông tin khách hàng
        customer_data = []
        for i in range(5):
            st.write(f"Khách hàng {i+1}")
            recency = st.slider("Recency (ngày)", 1, 365, 100, key=f"recency_{i}")
            frequency = st.slider("Frequency (đơn hàng)", 1, 50, 5, key=f"frequency_{i}")
            monetary = st.slider("Monetary ($)", 1, 1000, 100, key=f"monetary_{i}")
            customer_data.append({"Recency": recency, "Frequency": frequency, "Monetary": monetary})

        # Chuyển danh sách thành DataFrame
        df_customer = pd.DataFrame(customer_data)          
        # Thực hiện phân cụm khách hàng dựa trên giá trị của 3 cột này
        if not df_customer.empty:
            st.write("Phân cụm khách hàng...")
            # Dự đoán cụm bằng pipeline
            clusters = pipeline.predict(df_customer)
            # Thêm cột 'Kmeans_RFM' vào DataFrame
            df_customer['Kmeans_RFM'] = [cluster_to_group[cluster] for cluster in clusters]
        # In kết quả ra màn hình
        st.write("##### 3. Phân cụm khách hàng")
        st.write(df_customer)
        # Từ kết quả phân cụm khách hàng, người dùng có thể xem thông tin chi tiết của từng cụm khách hàng, xem biểu đồ, thống kê...
        # hoặc thực hiện các xử lý khác
    elif type == "Tải file Excel/CSV":
        # Nếu người dùng chọn tải file Excel/CSV
        st.subheader("Tải file Excel hoặc CSV")
        # Tạo file mẫu để tải về
        sample_df = pd.DataFrame(columns=['Member_number', 'Recency', 'Frequency', 'Monetary'])
        # Chuyển DataFrame thành CSV buffer
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        # Chuyển DataFrame thành Excel buffer
        excel_buffer = io.BytesIO()
        sample_df.to_excel(excel_buffer, index=False)
        excel_data = excel_buffer.getvalue()
        # Thêm nút tải file mẫu (CSV và Excel)
        st.write("Tải file mẫu để điền dữ liệu:")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Tải file CSV mẫu",
                data=csv_data,
                file_name="customer_data_template.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Tải file Excel mẫu",
                data=excel_data,
                file_name="customer_data_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )   
        # Upload file
        uploaded_file = st.file_uploader("Chọn file Excel hoặc CSV", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                # Đọc file dựa trên định dạng
                if uploaded_file.name.endswith('.csv'):
                    df_uploaded = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df_uploaded = pd.read_excel(uploaded_file)
                # Hiển thị dữ liệu ban đầu
                st.write("##### 2. Dữ liệu từ file tải lên")
                st.write(df_uploaded)
                # Kiểm tra các cột cần thiết
                required_columns = ['Recency', 'Frequency', 'Monetary']
                if all(col in df_uploaded.columns for col in required_columns):
                    # Dự đoán cụm
                    clusters = pipeline.predict(df_uploaded[required_columns])
                    df_uploaded['Kmeans_RFM'] = [cluster_to_group[cluster] for cluster in clusters]
                    # Hiển thị kết quả phân cụm
                    st.write("##### 3. Kết quả phân cụm khách hàng")
                    st.write(df_uploaded)
                else:
                    st.error("File tải lên cần có các cột: Recency, Frequency, Monetary")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi xử lý file: {str(e)}")
        else:
            st.info("Vui lòng tải lên một file Excel hoặc CSV để bắt đầu.")