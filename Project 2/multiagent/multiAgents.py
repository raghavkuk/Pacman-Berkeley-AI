# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        nearestGhostDist = 1234
        for ghost in newGhostStates:
            ghostDist = manhattanDistance(newPos, ghost.getPosition())
            if ghostDist < nearestGhostDist:
                nearestGhostDist = ghostDist


        if nearestGhostDist != 0:
            ghostWeight = 20*(1/nearestGhostDist)
        else:
            ghostWeight = 1000

        nearestFoodDist = 5000
        for food in newFood.asList():
            foodDist = manhattanDistance(newPos, food)
            if foodDist < nearestFoodDist:
                nearestFoodDist = foodDist

        if nearestFoodDist == 5000:
            nearestFoodDist = 0

        return -500*len(newFood.asList()) - 10*nearestFoodDist - 5*ghostWeight


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        actionList = self.minimax(gameState, 0, 0)
        lastAction = actionList[-1]
        return lastAction

    def minimax(self, gameState, agent, depth):

        resultList = []

        if depth >= self.depth: 
            resultList.append(self.evaluationFunction(gameState))
            return resultList
 
        else:  
            nextPossibleStates = []
            for action in gameState.getLegalActions(agent):
                stateTuple = (gameState.generateSuccessor(agent, action), action)
                nextPossibleStates.append(stateTuple)
        
            if not nextPossibleStates:
                nextPossibleStates.append((gameState, None))
            nextAgent = (agent + 1) % gameState.getNumAgents()
            ghostIdentifier = nextAgent == 0

            if agent == 0: 
                maxList = []
                for state, action in nextPossibleStates:
                    maxList.append(self.minimax(state, nextAgent, depth) + [action])
                return max(maxList)
            else:
                minList = []
                for state, action in nextPossibleStates:
                    minList.append(self.minimax(state, nextAgent, depth+ghostIdentifier) + [action])
                return min(minList)
           
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actionList = self.alphabeta(gameState, 0, 0, float("-inf"), float("inf"))
        lastAction = actionList[-1]
        return lastAction

    def alphabeta(self, gameState, agent, depth, alpha, beta):

        if depth >= self.depth or gameState.isWin() or gameState.isLose(): 
            resultList = []
            resultList.append(self.evaluationFunction(gameState))
            return resultList
 
        else:  
            nextPossibleActions = gameState.getLegalActions(agent)
            if not nextPossibleActions: nextPossibleActions = (gameState, None)

            nextAgent = (agent + 1) % gameState.getNumAgents()
            ghostIdentifier = nextAgent == 0

            if agent == 0: 
                bestOption = [float("-inf"), None]
                for action in nextPossibleActions:
                    nextState = gameState.generateSuccessor(agent, action)
                    nextStateValue = self.alphabeta(nextState, nextAgent, depth, alpha, beta) + [action]
                    bestOption = max(bestOption, nextStateValue)
                    curValue = bestOption[0]
                    if curValue > beta:
                        return bestOption
                    alpha = max(alpha, curValue)
            else:
                bestOption = [float("inf"), None]
                for action in nextPossibleActions:
                    nextState = gameState.generateSuccessor(agent, action)
                    nextStateValue = self.alphabeta(nextState, nextAgent, depth+ghostIdentifier, alpha, beta) + [action]
                    bestOption = min(bestOption, nextStateValue)
                    curValue = bestOption[0]
                    if curValue < alpha:
                        return bestOption
                    beta = min(beta, curValue)
            return bestOption


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actionList = self.expectimax(gameState, 0, 0)
        lastAction = actionList[-1]
        return lastAction

    def expectimax(self, gameState, agent, depth):

        resultList = []

        if depth >= self.depth: 
            resultList.append(self.evaluationFunction(gameState))
            return resultList
 
        else:  
            nextPossibleStates = []
            for action in gameState.getLegalActions(agent):
                stateTuple = (gameState.generateSuccessor(agent, action), action)
                nextPossibleStates.append(stateTuple)
        
            if not nextPossibleStates:
                nextPossibleStates.append((gameState, None))
            nextAgent = (agent + 1) % gameState.getNumAgents()
            ghostIdentifier = nextAgent == 0

            if agent == 0: 
                maxList = []
                for state, action in nextPossibleStates:
                    maxList.append(self.expectimax(state, nextAgent, depth) + [action])
                return max(maxList)
            else:
                expectiList = []
                for state, action in nextPossibleStates:
                    expectiList.append(self.expectimax(state, nextAgent, depth+ghostIdentifier) + [action])

                total = 0
                for value in expectiList:
                    total += value[0]

                return [total/float(len(expectiList))]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    curPostion = currentGameState.getPacmanPosition()

    ghostDistances = []
    for ghost in currentGameState.getGhostPositions():
        ghostDistances.append(manhattanDistance(curPostion, ghost))

    if min(ghostDistances) != 0:
        nearestGhostDistance = 1.0/min(ghostDistances)
    else:
        nearestGhostDistance = 1.0

    capsulesEaten = len(currentGameState.getCapsules())
    capsuleWeight = 1.0/(capsulesEaten+1)

    return currentGameState.getScore() + nearestGhostDistance*20 + capsuleWeight*500

# Abbreviation
better = betterEvaluationFunction

