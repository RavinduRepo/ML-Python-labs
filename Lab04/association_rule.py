from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# 1. Import the provided groceries.csv dataset.
groceries_df = pd.read_csv('./groceries.csv')

# 2. Explore the dataset
print("Number of transactions:", len(groceries_df))
print("Sample transactions:")
print(groceries_df.head())

# 3. Build the frequent-item DataFrame
transactions = groceries_df.apply(lambda row: [item for item in row if pd.notnull(item)], axis=1).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
groceries_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# 4. Apply Apriori to find itemsets with support > 8%
freq_itemsets = apriori(groceries_onehot, min_support=0.08, use_colnames=True)
print("\nFrequent itemsets (support > 8%):")
print(freq_itemsets)

# 5. Generate association rules using the lift metric
rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
print("\nAssociation rules (lift >= 1):")
print(rules)

# 6. Select one rule and interpret it
if not rules.empty:
    selected_rule = rules.iloc[0]
    print("\nSelected Rule:")
    print(selected_rule)
    print("\nInterpretation:")
    print(f"If a customer buys {list(selected_rule['antecedents'])}, they are likely to also buy {list(selected_rule['consequents'])}.")
    print(f"This rule has a confidence of {selected_rule['confidence']:.2f} and a lift of {selected_rule['lift']:.2f}, meaning the likelihood of buying {list(selected_rule['consequents'])} increases by a factor of {selected_rule['lift']:.2f} when {list(selected_rule['antecedents'])} is purchased.")

# 7. How many rules satisfy both lift > 4 and confidence > 0.8?
filtered_rules = rules[(rules['lift'] > 4) & (rules['confidence'] > 0.8)]
print("\nNumber of rules with lift > 4 and confidence > 0.8:", filtered_rules.shape[0])
print(filtered_rules)