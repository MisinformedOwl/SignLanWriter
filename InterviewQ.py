"""
NOT IMPORTANT: PLEASE IGNORE
This piece of code was for a job interview question. I made it in this file, and so i might aswell upload it with it.
"""

import requests                      #Requests is required for URL processing
from bs4 import BeautifulSoup as bs  #bs4 is helpful for navigation and selection
import pandas as pd

"""
Responcible for grabbing the data and putting it into a dataframe

parameters:
url: the url of the secret message

Returns: Pandas DataFrame
"""
def grabData(url: str) -> (pd.DataFrame, int):
    doc = requests.get(url)
    soup = bs(doc.content, "html.parser")

    xcords = []
    ycords = []
    char   = []

    table = soup.find("table") #Find the table so we have an isolated environment
    
    largestY = 0 #Since for some reason 0,0 is the bottom left cordinate we have to find the highest Y so we can work backwards when printing.

    for row in table.find_all("tr")[1:]: #Skip the header so we only have the values
        columns = row.find_all("td")
        xcords.append(int(columns[0].text))
        char.append(columns[1].text)
        y = int(columns[2].text)
        largestY = max(y,largestY)
        ycords.append(y)

    data = pd.DataFrame({"x":xcords, "y":ycords, "char":char})
    data = data.sort_values(by=["y","x"])
    return (data, largestY)
    
"""
This builds the character from the supplied dataframe

Parameters:
data: the dataframe with columns ["x", "y", "char"]
largestY: The largest y value since the 0,0 cordinate is in the bottom left instead of the presumed top left.
"""
def build(data : pd.DataFrame, largestY: int) -> None:
    for y in range(largestY,-1,-1): # for y to 0
        row = data[data["y"] == y]
        col = 0 # Start from the start (also helps to see if a space is needed)
        for _, r in row.iterrows():
            x = r["x"]
            diff = x-col
            col = x
            print(f"{" "*(diff-1)}{r["char"]}", end="")
        print("")

if __name__ == "__main__":
    data, largest = grabData("https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub")
    build(data, largest)