import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better readability
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load data
df = pd.read_csv('books_data_20251202_231851.csv')

# Convert numeric columns
df['retail_price'] = pd.to_numeric(df['retail_price'], errors='coerce')
df['old_price'] = pd.to_numeric(df['old_price'], errors='coerce')
df['rating_value'] = pd.to_numeric(df['rating_value'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
df['seller_rating'] = pd.to_numeric(df['seller_rating'], errors='coerce')
df['max_installment_months'] = pd.to_numeric(df['max_installment_months'], errors='coerce')

print("Generating business insights charts...")
print(f"Total records: {len(df)}")

# ===========================
# 1. PRICE DISTRIBUTION & STRATEGY
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📊 Pricing Strategy Analysis', fontsize=18, fontweight='bold')

# 1.1 Price Distribution
ax1 = axes[0, 0]
prices = df['retail_price'].dropna()
ax1.hist(prices[prices <= 50], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
ax1.axvline(prices.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {prices.median():.2f} AZN')
ax1.axvline(prices.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {prices.mean():.2f} AZN')
ax1.set_xlabel('Price (AZN)')
ax1.set_ylabel('Number of Books')
ax1.set_title('Price Distribution (Books ≤50 AZN)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 1.2 Discount Analysis
ax2 = axes[0, 1]
df['has_discount'] = df['old_price'] > df['retail_price']
df['discount_percent'] = ((df['old_price'] - df['retail_price']) / df['old_price'] * 100).fillna(0)
discount_books = df[df['has_discount']]
discount_summary = {
    'Books with Discount': len(discount_books),
    'Books without Discount': len(df) - len(discount_books)
}
colors = ['#e74c3c', '#95a5a6']
wedges, texts, autotexts = ax2.pie(discount_summary.values(), labels=discount_summary.keys(),
                                     autopct='%1.1f%%', colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax2.set_title(f'Discount Availability\nAvg Discount: {discount_books["discount_percent"].mean():.1f}%')

# 1.3 Price Segments
ax3 = axes[1, 0]
price_segments = pd.cut(df['retail_price'], bins=[0, 5, 10, 20, 50, 1000],
                        labels=['Budget\n(0-5)', 'Economy\n(5-10)', 'Standard\n(10-20)',
                               'Premium\n(20-50)', 'Luxury\n(50+)'])
segment_counts = price_segments.value_counts().sort_index()
bars = ax3.bar(segment_counts.index, segment_counts.values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
ax3.set_ylabel('Number of Books')
ax3.set_title('Market Segmentation by Price')
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')

# 1.4 Top 10 Most Expensive Books
ax4 = axes[1, 1]
top_expensive = df.nlargest(10, 'retail_price')[['name', 'retail_price']].copy()
top_expensive['short_name'] = top_expensive['name'].str[:40] + '...'
bars = ax4.barh(range(len(top_expensive)), top_expensive['retail_price'], color='#e67e22')
ax4.set_yticks(range(len(top_expensive)))
ax4.set_yticklabels(top_expensive['short_name'], fontsize=8)
ax4.set_xlabel('Price (AZN)')
ax4.set_title('Top 10 Most Expensive Books')
ax4.invert_yaxis()
for i, (idx, row) in enumerate(top_expensive.iterrows()):
    ax4.text(row['retail_price'], i, f' {row["retail_price"]:.0f} AZN',
            va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/1_pricing_strategy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 1_pricing_strategy.png")
plt.close()

# ===========================
# 2. CATEGORY ANALYSIS
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📚 Category & Market Composition Analysis', fontsize=18, fontweight='bold')

# 2.1 Top 15 Categories
ax1 = axes[0, 0]
top_categories = df['category_name'].value_counts().head(15)
bars = ax1.barh(range(len(top_categories)), top_categories.values, color='#3498db')
ax1.set_yticks(range(len(top_categories)))
ax1.set_yticklabels(top_categories.index, fontsize=9)
ax1.set_xlabel('Number of Books')
ax1.set_title('Top 15 Categories by Volume')
ax1.invert_yaxis()
for i, v in enumerate(top_categories.values):
    ax1.text(v, i, f' {v} ({v/len(df)*100:.1f}%)', va='center', fontweight='bold')

# 2.2 Category by Average Price
ax2 = axes[0, 1]
category_prices = df.groupby('category_name')['retail_price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
top_price_cats = category_prices[category_prices['count'] >= 20].head(10)  # At least 20 books
bars = ax2.barh(range(len(top_price_cats)), top_price_cats['mean'], color='#2ecc71')
ax2.set_yticks(range(len(top_price_cats)))
ax2.set_yticklabels(top_price_cats.index, fontsize=9)
ax2.set_xlabel('Average Price (AZN)')
ax2.set_title('Top 10 Categories by Average Price\n(min 20 books)')
ax2.invert_yaxis()
for i, v in enumerate(top_price_cats['mean']):
    ax2.text(v, i, f' {v:.1f} AZN', va='center', fontweight='bold')

# 2.3 Market Share - Top Categories
ax3 = axes[1, 0]
top_5_cats = df['category_name'].value_counts().head(5)
others = df['category_name'].value_counts().iloc[5:].sum()
pie_data = list(top_5_cats.values) + [others]
pie_labels = list(top_5_cats.index) + ['Others']
colors_pie = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6']
wedges, texts, autotexts = ax3.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax3.set_title('Market Share: Top 5 Categories vs Others')

# 2.4 Books per Category Distribution
ax4 = axes[1, 1]
books_per_cat = df['category_name'].value_counts().values
ax4.hist(books_per_cat, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
ax4.axvline(np.median(books_per_cat), color='red', linestyle='--', linewidth=2,
           label=f'Median: {np.median(books_per_cat):.0f} books')
ax4.set_xlabel('Books per Category')
ax4.set_ylabel('Number of Categories')
ax4.set_title('Distribution of Books Across Categories')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/2_category_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 2_category_analysis.png")
plt.close()

# ===========================
# 3. SELLER ANALYSIS
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🏪 Seller Performance & Market Share Analysis', fontsize=18, fontweight='bold')

# 3.1 Top 10 Sellers by Volume
ax1 = axes[0, 0]
top_sellers = df['seller_name'].value_counts().head(10)
bars = ax1.barh(range(len(top_sellers)), top_sellers.values, color='#e67e22')
ax1.set_yticks(range(len(top_sellers)))
ax1.set_yticklabels(top_sellers.index, fontsize=9)
ax1.set_xlabel('Number of Books')
ax1.set_title('Top 10 Sellers by Catalog Size')
ax1.invert_yaxis()
for i, v in enumerate(top_sellers.values):
    ax1.text(v, i, f' {v} ({v/len(df)*100:.1f}%)', va='center', fontweight='bold', fontsize=9)

# 3.2 Seller Rating Distribution
ax2 = axes[0, 1]
seller_ratings = df.groupby('seller_name').agg({
    'seller_rating': 'first',
    'name': 'count'
}).rename(columns={'name': 'book_count'})
seller_ratings = seller_ratings[seller_ratings['book_count'] >= 10].sort_values('seller_rating', ascending=False).head(15)
bars = ax2.barh(range(len(seller_ratings)), seller_ratings['seller_rating'], color='#1abc9c')
ax2.set_yticks(range(len(seller_ratings)))
ax2.set_yticklabels(seller_ratings.index, fontsize=8)
ax2.set_xlabel('Seller Rating (%)')
ax2.set_title('Top 15 Sellers by Rating\n(min 10 books)')
ax2.invert_yaxis()
ax2.set_xlim(70, 100)
for i, v in enumerate(seller_ratings['seller_rating']):
    ax2.text(v, i, f' {v:.0f}%', va='center', fontweight='bold')

# 3.3 Market Concentration
ax3 = axes[1, 0]
top_3_sellers = df['seller_name'].value_counts().head(3)
top_10_sellers = df['seller_name'].value_counts().head(10).sum()
others = len(df) - top_10_sellers
market_data = {
    f'Top 3 Sellers': top_3_sellers.sum(),
    f'Other Top 10': top_10_sellers - top_3_sellers.sum(),
    'All Others': others
}
colors_market = ['#e74c3c', '#f39c12', '#95a5a6']
wedges, texts, autotexts = ax3.pie(market_data.values(), labels=market_data.keys(),
                                     autopct='%1.1f%%', colors=colors_market, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax3.set_title('Market Concentration Analysis')

# 3.4 Seller Performance: Rating vs Catalog Size
ax4 = axes[1, 1]
seller_perf = df.groupby('seller_name').agg({
    'seller_rating': 'first',
    'name': 'count'
}).rename(columns={'name': 'book_count'})
seller_perf = seller_perf[seller_perf['book_count'] >= 5]
scatter = ax4.scatter(seller_perf['book_count'], seller_perf['seller_rating'],
                      s=seller_perf['book_count']*2, alpha=0.6, c=seller_perf['seller_rating'],
                      cmap='RdYlGn', edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Catalog Size (Number of Books)')
ax4.set_ylabel('Seller Rating (%)')
ax4.set_title('Seller Performance: Rating vs Catalog Size\n(bubble size = catalog size)')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Rating %')

# Annotate top sellers
top_5 = seller_perf.nlargest(5, 'book_count')
for idx, row in top_5.iterrows():
    ax4.annotate(idx, (row['book_count'], row['seller_rating']),
                fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('charts/3_seller_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 3_seller_analysis.png")
plt.close()

# ===========================
# 4. RATING & CUSTOMER SATISFACTION
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('⭐ Customer Ratings & Satisfaction Analysis', fontsize=18, fontweight='bold')

# 4.1 Rating Distribution
ax1 = axes[0, 0]
rated_books = df[df['rating_value'] > 0]
rating_counts = rated_books['rating_value'].value_counts().sort_index()
bars = ax1.bar(rating_counts.index, rating_counts.values, color='#f39c12', edgecolor='black')
ax1.set_xlabel('Rating (Stars)')
ax1.set_ylabel('Number of Books')
ax1.set_title(f'Rating Distribution\n({len(rated_books)} rated books out of {len(df)} total)')
ax1.set_xticks([1, 2, 3, 4, 5])
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(rated_books)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')

# 4.2 Review Engagement
ax2 = axes[0, 1]
rating_status = {
    'Rated Books': len(rated_books),
    'Unrated Books': len(df) - len(rated_books)
}
colors_rating = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax2.pie(rating_status.values(), labels=rating_status.keys(),
                                     autopct='%1.1f%%', colors=colors_rating, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax2.set_title('Customer Review Engagement')

# 4.3 Most Reviewed Books (Top 15)
ax3 = axes[1, 0]
most_reviewed = df.nlargest(15, 'rating_count')[['name', 'rating_count', 'rating_value']].copy()
most_reviewed['short_name'] = most_reviewed['name'].str[:35] + '...'
bars = ax3.barh(range(len(most_reviewed)), most_reviewed['rating_count'])
# Color by rating
colors_bars = plt.cm.RdYlGn(most_reviewed['rating_value'] / 5.0)
for i, (bar, color) in enumerate(zip(bars, colors_bars)):
    bar.set_color(color)
ax3.set_yticks(range(len(most_reviewed)))
ax3.set_yticklabels(most_reviewed['short_name'], fontsize=8)
ax3.set_xlabel('Number of Reviews')
ax3.set_title('Top 15 Most Reviewed Books\n(color: rating quality)')
ax3.invert_yaxis()
for i, row in most_reviewed.iterrows():
    idx = list(most_reviewed.index).index(i)
    ax3.text(row['rating_count'], idx, f' {int(row["rating_count"])} (★{row["rating_value"]:.1f})',
            va='center', fontsize=7)

# 4.4 Price vs Rating Correlation
ax4 = axes[1, 1]
price_rating_df = df[(df['rating_value'] > 0) & (df['retail_price'] <= 50)].copy()
scatter = ax4.scatter(price_rating_df['retail_price'], price_rating_df['rating_value'],
                     s=price_rating_df['rating_count']*3, alpha=0.5,
                     c=price_rating_df['rating_value'], cmap='RdYlGn',
                     edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Price (AZN)')
ax4.set_ylabel('Rating (Stars)')
ax4.set_title('Price vs Rating Correlation\n(bubble size = review count)')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 5.5)
plt.colorbar(scatter, ax=ax4, label='Rating')

# Add trend line
z = np.polyfit(price_rating_df['retail_price'], price_rating_df['rating_value'], 1)
p = np.poly1d(z)
ax4.plot(price_rating_df['retail_price'].sort_values(),
        p(price_rating_df['retail_price'].sort_values()),
        "r--", alpha=0.8, linewidth=2, label='Trend')
ax4.legend()

plt.tight_layout()
plt.savefig('charts/4_rating_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 4_rating_analysis.png")
plt.close()

# ===========================
# 5. INSTALLMENT & PAYMENT OPTIONS
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('💳 Installment Plans & Payment Options Analysis', fontsize=18, fontweight='bold')

# 5.1 Installment Availability
ax1 = axes[0, 0]
installment_data = df['installment_enabled'].value_counts()
labels = ['Installment Available', 'No Installment']
colors_inst = ['#27ae60', '#e74c3c']
wedges, texts, autotexts = ax1.pie(installment_data.values, labels=labels,
                                     autopct='%1.1f%%', colors=colors_inst, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax1.set_title('Installment Payment Availability')

# 5.2 Installment Period Distribution
ax2 = axes[0, 1]
installment_books = df[df['installment_enabled'] == True]
installment_months = installment_books['max_installment_months'].value_counts().sort_index()
bars = ax2.bar(installment_months.index, installment_months.values, color='#3498db', edgecolor='black')
ax2.set_xlabel('Maximum Installment Months')
ax2.set_ylabel('Number of Books')
ax2.set_title(f'Installment Period Distribution\n({len(installment_books)} books with installment)')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

# 5.3 Price Range by Installment Availability
ax3 = axes[1, 0]
inst_yes = df[df['installment_enabled'] == True]['retail_price'].dropna()
inst_no = df[df['installment_enabled'] == False]['retail_price'].dropna()
box_data = [inst_yes[inst_yes <= 100], inst_no[inst_no <= 100]]
bp = ax3.boxplot(box_data, labels=['With Installment', 'Without Installment'],
                 patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], ['#27ae60', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Price (AZN)')
ax3.set_title('Price Distribution by Installment Availability\n(books ≤100 AZN)')
ax3.grid(True, alpha=0.3, axis='y')

# Add statistics
ax3.text(1, inst_yes[inst_yes <= 100].median(), f'Median: {inst_yes[inst_yes <= 100].median():.1f}',
        ha='center', va='bottom', fontweight='bold', color='darkgreen')
ax3.text(2, inst_no[inst_no <= 100].median(), f'Median: {inst_no[inst_no <= 100].median():.1f}',
        ha='center', va='bottom', fontweight='bold', color='darkred')

# 5.4 Installment Options by Category (Top 10)
ax4 = axes[1, 1]
cat_installment = df[df['installment_enabled'] == True].groupby('category_name').size().sort_values(ascending=False).head(10)
bars = ax4.barh(range(len(cat_installment)), cat_installment.values, color='#9b59b6')
ax4.set_yticks(range(len(cat_installment)))
ax4.set_yticklabels(cat_installment.index, fontsize=9)
ax4.set_xlabel('Books with Installment')
ax4.set_title('Top 10 Categories with Installment Options')
ax4.invert_yaxis()
for i, v in enumerate(cat_installment.values):
    total_in_cat = df[df['category_name'] == cat_installment.index[i]].shape[0]
    percentage = (v/total_in_cat)*100
    ax4.text(v, i, f' {v} ({percentage:.0f}%)', va='center', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig('charts/5_installment_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 5_installment_analysis.png")
plt.close()

# ===========================
# 6. BRAND ANALYSIS
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🏷️ Brand Analysis & Market Presence', fontsize=18, fontweight='bold')

# 6.1 Top 15 Brands
ax1 = axes[0, 0]
top_brands = df['brand'].value_counts().head(15)
bars = ax1.barh(range(len(top_brands)), top_brands.values, color='#16a085')
ax1.set_yticks(range(len(top_brands)))
ax1.set_yticklabels(top_brands.index, fontsize=9)
ax1.set_xlabel('Number of Books')
ax1.set_title('Top 15 Brands by Book Count')
ax1.invert_yaxis()
for i, v in enumerate(top_brands.values):
    ax1.text(v, i, f' {v} ({v/len(df)*100:.1f}%)', va='center', fontweight='bold', fontsize=8)

# 6.2 Brand Diversity
ax2 = axes[0, 1]
total_brands = df['brand'].nunique()
no_brand = (df['brand'] == 'No Brand').sum()
branded = len(df) - no_brand
brand_data = {
    'Branded Books': branded,
    'No Brand': no_brand
}
colors_brand = ['#2ecc71', '#95a5a6']
wedges, texts, autotexts = ax2.pie(brand_data.values(), labels=brand_data.keys(),
                                     autopct='%1.1f%%', colors=colors_brand, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax2.set_title(f'Brand Presence\nTotal Unique Brands: {total_brands}')

# 6.3 Average Price by Top Brands
ax3 = axes[1, 0]
brand_avg_price = df.groupby('brand').agg({
    'retail_price': 'mean',
    'name': 'count'
}).rename(columns={'name': 'count'})
brand_avg_price = brand_avg_price[brand_avg_price['count'] >= 15].sort_values('retail_price', ascending=False).head(10)
bars = ax3.barh(range(len(brand_avg_price)), brand_avg_price['retail_price'], color='#e67e22')
ax3.set_yticks(range(len(brand_avg_price)))
ax3.set_yticklabels(brand_avg_price.index, fontsize=9)
ax3.set_xlabel('Average Price (AZN)')
ax3.set_title('Top 10 Brands by Average Price\n(min 15 books)')
ax3.invert_yaxis()
for i, v in enumerate(brand_avg_price['retail_price']):
    ax3.text(v, i, f' {v:.1f} AZN', va='center', fontweight='bold')

# 6.4 Brand Rating Performance
ax4 = axes[1, 1]
brand_ratings = df[df['rating_value'] > 0].groupby('brand').agg({
    'rating_value': 'mean',
    'rating_count': 'sum',
    'name': 'count'
}).rename(columns={'name': 'book_count'})
brand_ratings = brand_ratings[brand_ratings['book_count'] >= 10].sort_values('rating_value', ascending=False).head(12)
bars = ax4.barh(range(len(brand_ratings)), brand_ratings['rating_value'], color='#f39c12')
ax4.set_yticks(range(len(brand_ratings)))
ax4.set_yticklabels(brand_ratings.index, fontsize=8)
ax4.set_xlabel('Average Rating')
ax4.set_title('Top 12 Brands by Average Rating\n(min 10 books)')
ax4.invert_yaxis()
ax4.set_xlim(0, 5.5)
for i, v in enumerate(brand_ratings['rating_value']):
    ax4.text(v, i, f' ★{v:.2f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/6_brand_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 6_brand_analysis.png")
plt.close()

# ===========================
# 7. COMPREHENSIVE DASHBOARD
# ===========================
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
fig.suptitle('📊 Books Market - Comprehensive Dashboard', fontsize=20, fontweight='bold')

# Key Metrics
ax_metrics = fig.add_subplot(gs[0, :])
ax_metrics.axis('off')
metrics_text = f"""
KEY BUSINESS METRICS

Total Books: {len(df):,} | Total Categories: {df['category_name'].nunique()} | Total Brands: {df['brand'].nunique()} | Total Sellers: {df['seller_name'].nunique()}

Average Price: {df['retail_price'].mean():.2f} AZN | Median Price: {df['retail_price'].median():.2f} AZN | Price Range: {df['retail_price'].min():.2f} - {df['retail_price'].max():.2f} AZN

Books with Discount: {(df['has_discount'].sum()/len(df)*100):.1f}% | Avg Discount: {discount_books['discount_percent'].mean():.1f}%

Installment Available: {(df['installment_enabled'].sum()/len(df)*100):.1f}% | Rated Books: {(len(rated_books)/len(df)*100):.1f}% | Avg Rating: {rated_books['rating_value'].mean():.2f}/5.0
"""
ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Mini charts
# 1. Top 5 Categories
ax1 = fig.add_subplot(gs[1, 0])
top_5_cats = df['category_name'].value_counts().head(5)
ax1.barh(range(len(top_5_cats)), top_5_cats.values, color='#3498db')
ax1.set_yticks(range(len(top_5_cats)))
ax1.set_yticklabels([name[:20] for name in top_5_cats.index], fontsize=8)
ax1.set_title('Top 5 Categories', fontsize=10, fontweight='bold')
ax1.invert_yaxis()

# 2. Top 5 Sellers
ax2 = fig.add_subplot(gs[1, 1])
top_5_sellers = df['seller_name'].value_counts().head(5)
ax2.barh(range(len(top_5_sellers)), top_5_sellers.values, color='#e67e22')
ax2.set_yticks(range(len(top_5_sellers)))
ax2.set_yticklabels([name[:20] for name in top_5_sellers.index], fontsize=8)
ax2.set_title('Top 5 Sellers', fontsize=10, fontweight='bold')
ax2.invert_yaxis()

# 3. Price Distribution
ax3 = fig.add_subplot(gs[1, 2])
ax3.hist(df['retail_price'][df['retail_price'] <= 50], bins=30, color='#2ecc71', alpha=0.7)
ax3.set_xlabel('Price (AZN)', fontsize=8)
ax3.set_ylabel('Count', fontsize=8)
ax3.set_title('Price Distribution', fontsize=10, fontweight='bold')

# 4. Rating Distribution
ax4 = fig.add_subplot(gs[1, 3])
rating_counts = rated_books['rating_value'].value_counts().sort_index()
ax4.bar(rating_counts.index, rating_counts.values, color='#f39c12')
ax4.set_xlabel('Rating', fontsize=8)
ax4.set_ylabel('Count', fontsize=8)
ax4.set_title('Rating Distribution', fontsize=10, fontweight='bold')
ax4.set_xticks([1, 2, 3, 4, 5])

# 5. Market Segments
ax5 = fig.add_subplot(gs[2, 0])
segment_counts.plot(kind='pie', ax=ax5, autopct='%1.0f%%', colors=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
ax5.set_ylabel('')
ax5.set_title('Price Segments', fontsize=10, fontweight='bold')

# 6. Installment
ax6 = fig.add_subplot(gs[2, 1])
installment_data.plot(kind='pie', ax=ax6, autopct='%1.0f%%', colors=['#27ae60', '#e74c3c'],
                     labels=['Available', 'Not Available'])
ax6.set_ylabel('')
ax6.set_title('Installment Options', fontsize=10, fontweight='bold')

# 7. Top Brands
ax7 = fig.add_subplot(gs[2, 2])
top_5_brands = df['brand'].value_counts().head(5)
ax7.barh(range(len(top_5_brands)), top_5_brands.values, color='#16a085')
ax7.set_yticks(range(len(top_5_brands)))
ax7.set_yticklabels([name[:15] for name in top_5_brands.index], fontsize=8)
ax7.set_title('Top 5 Brands', fontsize=10, fontweight='bold')
ax7.invert_yaxis()

# 8. Seller Performance
ax8 = fig.add_subplot(gs[2, 3])
seller_perf_top = seller_perf.nlargest(20, 'book_count')
ax8.scatter(seller_perf_top['book_count'], seller_perf_top['seller_rating'],
           s=seller_perf_top['book_count'], alpha=0.6, c=seller_perf_top['seller_rating'],
           cmap='RdYlGn')
ax8.set_xlabel('Catalog Size', fontsize=8)
ax8.set_ylabel('Rating %', fontsize=8)
ax8.set_title('Seller Performance', fontsize=10, fontweight='bold')
ax8.grid(True, alpha=0.3)

plt.savefig('charts/7_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 7_comprehensive_dashboard.png")
plt.close()

print("\n" + "="*60)
print("✅ All charts generated successfully!")
print("="*60)
print(f"Charts saved to: ./charts/")
print(f"Total charts: 7")
