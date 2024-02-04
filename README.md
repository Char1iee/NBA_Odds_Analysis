# cse151a_project
-link to notebook
-story behind data, previous work
-motivations behind the project
-objectives of the project
-broader impact of the project
-how we preprocessed (choosing model, over/underfitting, etc)
-conclusions
-possible future follow ups
-list of group members and contributions

## website
https://www.kaggle.com/datasets/christophertreasure/nba-odds-data


## data description
Date: The date on which the game took place
Season: The year/season during which the game took place
Team: The team on which the bet is being placed.
Home/Visitor: Boolean value that represents whether the team on which the bet is placed, is playing at their homeground, or is visiting their opponent.
Opponent: The team against which our team is playing.
Score: Our team's final game score
Opponent Score: The enemy team's final game score.
Moneyline bet: The simplest type of bet; if your team wins, you win the bet, no matter by how much (margin) they win. The moneyline payouts are an integer representing the payout if a given team wins; negative for favorites "likely" to win, positive for "underdogs" likely to lose. If negative represents how much money you must bet to gain $100, if positive represents how much you will gain on a $100 bet. For example, for teams A and B with moneylines -550 (underdog) and -800 (favorite) respectively: if you bet $100 on A and win you will gain $550, if you bet 800 on B you will gain $100 

total: Total points bet; Also known as over/under bet; this involves predicting if the total points socred by both teams will be over or under the projected point total, which is set by the sportsbook; assume the projected point total is 212, and you bet the final total points will be over the projected point total. If the actual points total after the game is higher than 212, then you win. If it is lower than 212, then you lose. If the actual score is equal to 212, then the bet is refunded.
spread: Point spread bet; This bet requires a team to win or lose by a specific number of points, which is set by the sportsbook. For example, if the spread is +3 for Los Angeles Lakers in a game that Lakers vs Clippers, and you bet Lakers will win, it means that you are betting on the chance that Lakers will win the game by at least 4 points more than Clippers
secondHalfTotal: 

## Data distributions - interpretation:

### season:
The data that we have goes from the 2008 season to the 2023 season, and is roughly evenly distributed to have the same number of games per season. This makes sense since the number of games played each season should be pretty similar if no rule changes were made.

### score: 
The scores seem to be somewhat normally distributed with the mean and median both being 104. The scores range from 54 to 168, and the standard deviation is 13.30. 

### opponentScore:
As opponentScore and score have the same data points just ordered differently, their distributions are identical.

### moneyLine:
moneyLine seems to have extreme outliers on both sides, as it ranges from -13000 to 6500, but 50% of the data lies between -240 to 195 and 90% of the data lies between -800 and 575.

### opponentMoneyLine:
As opponentMoneyLine and moneyLine have the same data points just ordered differently, their distributions are identical.

### total


### spread


### secondHalfSpread


## data preprocessing
To preprocess the data, we plan to one-hot encode the home/visitor column to make the data numerical, and convert the data values to be integers instead of objects. Based on our exploratory analysis, we found that the data in ______ column is roughly normal, and may benefit from standardization.

