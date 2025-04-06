import pandas as pd
import matplotlib.pyplot as plt




# #############################################################################
# LOAD DATA
# #############################################################################
country = "IT"
df_matcherIT = pd.read_excel("PriorityModelAssignment.xlsx", sheet_name=country)
df_matcherIT = df_matcherIT[["Customer", "Contribution", "modelType"]]
df_matcherIT = df_matcherIT.dropna()
df_matcherIT = df_matcherIT.sort_values(by='Contribution', ascending=False)
df_matcherIT['Cumulative_Contribution'] = df_matcherIT['Contribution'].cumsum()

country = "ES"
df_matcherES = pd.read_excel("PriorityModelAssignment.xlsx", sheet_name=country)
df_matcherES = df_matcherES[["Customer", "Contribution", "modelType"]]
df_matcherES = df_matcherES.dropna()
df_matcherES = df_matcherES.sort_values(by='Contribution', ascending=False)
df_matcherES['Cumulative_Contribution'] = df_matcherES['Contribution'].cumsum()

top_n_ES = 20
df_matcherES_top = df_matcherES.head(top_n_ES)
top_n_IT = 20
df_matcherIT_top = df_matcherIT.head(top_n_IT)

model_type_counts_ES = df_matcherES.groupby("modelType").size()
model_type_counts_IT = df_matcherIT.groupby("modelType").size()

model_type_counts_ES_sorted = model_type_counts_ES.sort_values(ascending=False)
model_type_counts_IT_sorted = model_type_counts_IT.sort_values(ascending=False)

contribution_sum_by_model_ES = df_matcherES.groupby("modelType")["Contribution"].sum()
contribution_sum_by_model_ES = contribution_sum_by_model_ES.sort_values(ascending=False)

contribution_sum_by_model_IT = df_matcherIT.groupby("modelType")["Contribution"].sum()
contribution_sum_by_model_IT = contribution_sum_by_model_IT.sort_values(ascending=False)





# #############################################################################
# FIGURE 1: COMPANY PORTFOLIO CONTRIBUTION
# #############################################################################
plt.figure(figsize=(12, 6))
plt.suptitle("Company Portfolio Contribution", fontweight="bold")

# Plot for Italy
plt.subplot(1, 2, 1)
plt.title("Portfolio Italy (Top "+str(top_n_IT)+")")
plt.bar(df_matcherIT_top["Customer"], df_matcherIT_top["Cumulative_Contribution"], color=(255/255, 84/255, 00))
plt.xticks(rotation=90)  # Rotate customer names for better readability
plt.ylabel("Cumulative Contribution")
plt.xlabel("Companies")
plt.ylim(0,100)

# Plot for Spain
plt.subplot(1, 2, 2)
plt.title("Portfolio Spain (Top "+str(top_n_ES)+")")
plt.bar(df_matcherES_top["Customer"], df_matcherES_top["Cumulative_Contribution"], color=(255/255, 84/255, 00))
plt.xticks(rotation=90)  # Rotate customer names for better readability
plt.ylabel("Cumulative Contribution")
plt.xlabel("Companies")
plt.ylim(0,100)
plt.tight_layout()
plt.show()




# #############################################################################
# FIGURE 2: COMPANY SIZE COMPARISON
# #############################################################################
plt.figure(figsize=(12, 6))
plt.suptitle("Company Size Comparison", fontweight="bold")

# Plot for Italy
plt.subplot(1, 2, 1)
plt.title("Portfolio Italy (Top "+str(top_n_IT)+")")
plt.boxplot(df_matcherIT["Contribution"]/7)
plt.gca().set_xticks([])
plt.ylabel("Energy Consumption [Daily Average, MWh]")

# Plot for Spain
plt.subplot(1, 2, 2)
plt.title("Portfolio Spain (Top "+str(top_n_ES)+")")
plt.boxplot(df_matcherES["Contribution"]/7)
plt.ylabel("Energy Consumption [Daily Average, MWh]")
plt.gca().set_xticks([])
plt.tight_layout()
plt.show()



# #############################################################################
# FIGURE 3: MODEL USAGE 
# #############################################################################
plt.figure(figsize=(12, 8))  # Adjust figure size for better spacing
plt.suptitle("Prediction Model Usage", fontweight="bold", fontsize=16)

# Plot for Italy (Model Type Counts)
plt.subplot(2, 2, 1)
plt.bar(model_type_counts_IT_sorted.index, model_type_counts_IT_sorted.values, color=(255/255, 84/255, 00))
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and adjust font size
plt.ylabel("# Customers", fontsize=12)
plt.title("Portfolio Italy (# Customers)", fontsize=14)

# Plot for Spain (Model Type Counts)
plt.subplot(2, 2, 2)
plt.bar(model_type_counts_ES_sorted.index, model_type_counts_ES_sorted.values, color=(255/255, 84/255, 00))
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and adjust font size
plt.ylabel("# Customers", fontsize=12)
plt.title("Portfolio Spain (# Customers)", fontsize=14)

# Plot for Italy (Contribution Sum by Model Type)
plt.subplot(2, 2, 3)
plt.bar(contribution_sum_by_model_IT.index, contribution_sum_by_model_IT.values, color=(199/255, 224/255, 237/255))
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and adjust font size
plt.ylabel("Electrical Power", fontsize=12)
# plt.title("Portfolio Italy (Electrical Power)", fontsize=14)

# Plot for Spain (Contribution Sum by Model Type)
plt.subplot(2, 2, 4)
plt.bar(contribution_sum_by_model_ES.index, contribution_sum_by_model_ES.values, color=(199/255, 224/255, 237/255))
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and adjust font size
plt.ylabel("Electrical Power", fontsize=12)
# plt.title("Portfolio Spain (Electrical Power)", fontsize=14)

# Adjust layout to prevent overlapping elements
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the suptitle
plt.show()