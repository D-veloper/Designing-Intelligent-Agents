from simpleBot2_withCounting_soln import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def runSetOfExperiments(numberOfRuns,numberOfBots):
    dirtCollectedList = []
    for _ in range(numberOfRuns):
        dirtCollectedList.append(runOneExperiment(numberOfBots))
    return dirtCollectedList
        
def runExperimentsWithDifferentParameters():
    resultsTable = {}
    for numberOfBots in range(1,11):
        dirtCollected = runSetOfExperiments(10,numberOfBots)
        resultsTable["robots: "+str(numberOfBots)] = dirtCollected
    results = pd.DataFrame(resultsTable)
    print(results)
    results.to_excel("data.xlsx")
    print(ttest_ind(results["robots: 1"],results["robots: 2"]))
    print(results.mean(axis=0))
    results.boxplot(grid=False)
    plt.show()


runExperimentsWithDifferentParameters()
