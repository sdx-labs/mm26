# March Machine Learning Madness Model

the underlying thesis of the machine learning modeling behind project is the "cinderalla system" - an attempt to create a "heat" variable which can be used in training to better increase the accuracy of the model. 

The "heat" variable is going to be used as a proxy for teams who are outperforming expectations. So first we must establish an "expectations" baseline by creating a holistic and thorough model, then compare that to the teams past performance on a 1 game, a 3 game, and a 5 game basis - teams who are "overperforming", or winning games that they have not been "expected" to win based on the original holistic model, should be assigned a "heat" variable which should have a significant impact on the outcome of the model.

Then we are going to run a model which performs many simulations of the current 2026 bracket (and all potential 2026 march madness matchups) - and compose a composite prediction, in accordance with Kaggle guidelines outlined in the .agents/kaggle/* folder. the composite prediction should be influenced by both the initial prediciton model, based on a derived ELO, and the "heat" variable, along with standard statistics included in the datasets available. Hopefully, the "heat" variable helps us predict some potential cinderalla's - across the many, many simulations, some teams may be consistently overperforming their expected outcome, and their heat variable would then rise, and influence their overall expected win probability.

rememebr that the end result of the prediction should be completely and totally optimized for the brier score, as outlined in the .agents/kaggle/ folders.

the "heat" variable is calculated in multiple parts:

1. the first is to create a model based on standard ELO - using only the data available in the kaggle data set, we are going to estimate the probabiltiy of a win using an XGBoost model based on stats available.
2. then we are going to compute an initial expected win probability for the low team to beat the high team.
3. then we are going to use this to create a 1 game, a 3 game, and a 5 game "heat" score which compares the teams actual performance over the last 5 games to the performance that was initially predicted.
4. then we are going to run many simulations which are going to result in an end predicition of the bracket, and a submission file acceptable for the rules and optimized for the brier score.