Description
Another year, another chance to predict the upsets, call the probabilities, and put your bracketology skills to the leaderboard test. In our twelfth annual March Machine Learning Mania competition, Kagglers will once again join the millions of fans who attempt to predict the outcomes of this year's college basketball tournaments. Unlike most fans, you will pick the winners and losers using a combination of rich historical data and computing power, while the ground truth unfolds on television.

You are provided data of historical NCAA games to forecast the outcomes of the Division 1 Men's and Women's basketball tournaments. This competition is the official 2026 edition, with points, medals, prizes, and basketball glory at stake.

We are continuing the format from last year where you are making predictions about every possible matchup in the tournament, evaluated using the Brier score. See the Evaluation Page for full details.

Prior to the start of the tournaments, the leaderboard of this competition will reflect scores from 2021-2025 only. Kaggle will periodically fill in the outcomes and rescore once the 2026 games begin.

Good luck and happy forecasting! basketball graph

Evaluation
Submissions are evaluated on the Brier score between the predicted probabilities and the actual game outcomes (this is equivalent to mean squared error in this context).

Submission File
As a reminder, the submission file format also has a revised format from prior iterations:

We have combined the Men's and Women's tournaments into one single competition. Your submission file should contain predictions for both.
You will be predicting the hypothetical results for every possible team matchup, not just teams that are selected for the NCAA tournament. This change was enacted to provide a longer time window to submit predictions for the 2026 tournament. Previously, the short time between Selection Sunday and the tournament tipoffs would require participants to quickly turn around updated predictions. By forecasting every possible outcome between every team, you can now submit a valid prediction at any point leading up to the tournaments.
You may submit as many times as you wish before the tournaments start, but make sure to select the one submission you want to count towards scoring. Do not rely on automatic selection to pick your submissions.
As with prior years, each game has a unique ID created by concatenating the season in which the game was played and the two team's respective TeamIds. For example, "2026_1101_1102" indicates a hypothetical matchup between team 1101 and 1102 in the year 2026. You must predict the probability that the team with the lower TeamId beats the team with the higher TeamId. Note that the men's teams and women's TeamIds do not overlap.

The resulting submission format looks like the following, where Pred represents the predicted probability that the first team will win:

ID,Pred
2026_1101_1102,0.5
2026_1101_1103,0.5
2026_1101_1104,0.5
...
Your 2026 submissions will score 0.0 if you have submitted predictions in the right format. The leaderboard of this competition will be only meaningful once the 2026 tournaments begin and Kaggle rescores your predictions!

Timeline
February 19, 2026 - Start Date
March 19, 2026 4PM UTC - Final Submission Deadline. Note that Kaggle will release updated data at least once in advance of the deadline in order to include as much of the current season's data as possible. 

Evaluation - Brier Score Notes
