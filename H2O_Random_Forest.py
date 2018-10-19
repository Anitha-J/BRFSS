# coding: utf-8

# In[29]:


import h2o
import math
from h2o.estimators import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch


# In[54]:


h2o.init(nthreads = -1, min_mem_size = 45)


# In[6]:


data = h2o.import_file('../data.csv')


# In[7]:


training_cols = ["deaf","blind","x.rfhlth","x.rfhype5","x.rfchol1","x.asthms1","x.drdxar1","x.race","x.age.g","x.bmi5cat","x.chldcnt","x.educag","x.incomg","x.smoker3","x.ecigsts","x.rfdrhv5","x.totinda"]


# In[11]:


data[training_cols] = data[training_cols].asfactor()
data["cvdinfr4"] = data["cvdinfr4"].asfactor()


# In[42]:


params1 = {'sample_rate': [x/100. for x in range(20,101)],
                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                'min_rows': [2**x for x in range(0,int(math.log(data.nrow,2)-1)+1)],
                'nbins': [2**x for x in range(4,11)],
                'nbins_cats': [2**x for x in range(4,13)],
                'min_split_improvement': [0,1e-8,1e-6,1e-4]
                             }


# In[43]:


search_criteria_tune = {
                   'seed' : 17,
                   'stopping_rounds' : 5,
                   'stopping_metric' : "AUC",
                   'stopping_tolerance': 1e-3
                   }


# In[44]:


drf = H2ORandomForestEstimator(nfolds = 10, stopping_rounds = 5, stopping_metric = "AUC",stopping_tolerance = 1e-4, balance_classes = True)


# In[51]:


final_grid = H2OGridSearch(drf, hyper_params = params1,grid_id = 'final_grid1')


# In[52]:


final_grid.train(x = training_cols, y = "cvdinfr4", training_frame = data)


# In[ ]:


sorted_final_grid = final_grid.get_grid(sort_by='auc',decreasing=True)

print(sorted_final_grid)


# In[ ]:


best_model = h2o.get_model(sorted_final_grid.sorted_metric_table()['model_ids'][0])
params_list = []
for key, value in best_model.params.iteritems():
    params_list.append(str(key)+" = "+str(value['actual']))
params_list
h2o.save_model(best_model, "bestModel.csv", force=True)


# In[ ]:


print("TOP 200 Models\n\n")

for i in range(200):
    print("Model "+str(i))
    best_model = h2o.get_model(sorted_final_grid.sorted_metric_table()['model_ids'][i])
    print(best_model)
    h2o.save_model(best_model, "bestModel"+str(i)+".csv", force=True)
    params_list = []
    for key, value in best_model.params.iteritems():
        params_list.append(str(key)+" = "+str(value['actual']))
    print(params_list)


# In[53]:




# In[ ]:




