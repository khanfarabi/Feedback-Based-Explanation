# Feedback-Based-Explanation

In this project, we are explaining the predictions made by Machine Learning Model using SHAP (https://github.com/slundberg/shap). Here we have tried to represent SHAP relational explanations. The Yelp review data is used in this experiment and the data is available in the link (https://drive.google.com/drive/folders/1o-UmrtdLdYVTvUhWd75khzSqppP1upPq?usp=sharing). Here 1000 reviews (500 positive reviews and 500negative reviews) have been used in the experiment. 


# Run the Code to get the Non_Relational Shap Explanation Accuracy for the multiple feedback with voting mechanism with Review Data:
1. In Non_Relational_SHAP_Accuracy_update_1 the following two commands need to run:


# Without Feedback Shap Explanation Accuracy

        without_feedback()
        
# With Feedback per cluster SHAP Explanation Accuracy

st=10 # Statring # of cluster
lm=31 # Limits of # of cluster
df=5 # difference between 2 cluster mechanism.
withfeedback(st,lm,df)
Here the values of the st,lm, and df can be changes. st defines the initial number of clusters, lm defines the highest number of clusters we want to check, df defines the difference between two cluster models, for example if we have initial cluster size st=5, the next cluster model will have st=10, because df is 5 here.

# Data

The feedbacks are in the data folder. The Yelp Hotel Review Data available in the above mentioned link.


# Demo output for the Non_Relational Shap Explanation Accuracy for the multiple feedback with voting mechanism with Review Data  update-1
  
  
  The Explanation Accuracy With Human Feedback: Here we have varied the clusters from, 10,15,20,25, and 30 and compute word explanation accuracy with the increase of the  clusters. The accuracy is computed considering both posotive and negative reviews, only considerating positive reviews, and only considering negative reviews. In this experiment 1000 reviews (500 positive, 500 negative) are used.
  
  
  
  
  
  # Considering Both Positive and Negative Reviews
     Number of Clusters    Explanation Accuracy
     10                    0.6358024691358024
     15                   0.6666666666666666
     20                    0.6979166666666667
     25                    0.7150537634408602
     30                    0.7857142857142857




   # Considering only Positive  Reviews
     Number of Clusters    Explanation Accuracy
     10                    0.6388888888888888
     15                    0.6166666666666666
     20                    0.6296296296296295
     25                    0.6666666666666666
     30                    1.0

      
      
      
   # Considering only Negative  Reviews
     Number of Clusters    Explanation Accuracy
     10                    0.611111111111111
     15                    0.8333333333333334
     20                    0.7857142857142857
     25                    0.8166666666666667
     30                    0.7
  
  
  
  
  # Explanation  for the above Non_Relational Shap Explanation Accuracy results
  
  If we observe the above explanation accuracy allong with the increase with the clusters, we see that, the explanation accuracy remains same even with increase of the clusters specifically in the case where we consider both the positive and negative reviews together. If we consider only positive reviews we see that for the clusters 15,25, and 35 the accuracy in increasing state;however for the clusters 45 and 55 the accuracy little bit low and in case of the negative reviews we see the variations of the acuuracy. This is happening because we are taking mulltiple feedbacks and based on the feedbacks we are generating new explanations. Here are the feedback is taking against the explanations of the original data and the user gives 1 if the explanation is good otherwise 0. Therefore, we keep the features (words) similar to the explanations ranked as 1 otherwise removed if it is 0. As the human opinion varies the new explanations are also varies. Because we are keeping similar word features using neural network embedding and removing word features based on the human feedback. That is why we see the variation in the accuracy as well. Further, we can say that, increasing the clustering will not improve the explanations always in prectice because human opinion has the influence on the explanation and the accuracy, and from the above results we can clude that at some point the explanation accuracy reaches to the saturation level along with the increase of the clusters in some cases.

            

   # The Explanation Accuracy Without  Human Feedback with Review Data: 
       #Considering Both Positive and Negative Reviews
         Explanation Accuracy:  0.29463507625272184
         
         
        #Considering only Positive  Reviews
         Explanation Accuracy:  0.32059925093633085
         
         
         #Considering only Negative  Reviews
         Explanation Accuracy:  0.25852864583333407
         
         
         
# Packages need to be installed


Python=3.6

Gensim=3.8

Jupyter Notebook






