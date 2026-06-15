"""
fiqa_glossary.py - LLM-distilled knowledge for fiqa's recurring finance terms.

Written by the teacher LLM from general financial knowledge (not the gold docs),
rich in the bridging vocabulary the relevant answer passages use. Each entry is
an editable, appendable fact - the LLM taught the lattice once; the lattice
serves it forever at sub-ms with no model in the loop.
"""

GLOSSARY = {
    "etf": "ETF exchange traded fund is a basket of securities such as stocks or "
           "bonds that trades on a stock exchange like a single share, offering "
           "diversification and low fees, similar to index mutual funds",
    "llc": "LLC limited liability company is a business legal structure that "
           "protects the owners personal assets from business debts, with pass "
           "through taxation reported on the personal income tax return",
    "irs": "IRS Internal Revenue Service is the United States federal tax agency "
           "that collects income tax and enforces tax law, processing tax returns "
           "deductions refunds and withholding",
    "brokerage": "a brokerage account is an investment account held with a broker "
                 "or brokerage firm used to buy and sell stocks bonds ETFs and "
                 "other securities",
    "debit": "a debit card withdraws money directly from a checking or bank "
             "account, unlike a credit card which borrows; a debit reduces the "
             "account balance immediately",
    "portfolio": "an investment portfolio is a collection of financial assets such "
                 "as stocks bonds ETFs mutual funds and cash held by an investor "
                 "for diversification and returns",
    "deductible": "a tax deductible expense reduces taxable income; deductions "
                  "lower the income tax owed, such as business expenses mortgage "
                  "interest or charitable donations",
    "deductions": "tax deductions are expenses subtracted from gross income to "
                  "reduce taxable income and tax owed, either the standard "
                  "deduction or itemized deductions",
    "estate": "estate is a person's total assets and property; estate tax applies "
              "to inherited wealth transferred at death, and real estate is land "
              "and property ownership",
    "cap": "market cap or capitalization is the total value of a company's shares, "
           "equal to share price times shares outstanding, classifying firms as "
           "large mid or small cap",
    "exchange": "a stock exchange such as NYSE or NASDAQ is a marketplace where "
                "securities stocks bonds and ETFs are listed bought sold and traded",
    "traded": "trading is buying and selling securities such as stocks bonds and "
              "ETFs on an exchange or through a broker",
    "employed": "a self employed individual works for themselves rather than an "
                "employer, paying self employment tax and reporting business "
                "income and expenses on a tax return",
    "accounting": "accounting is the recording and reporting of financial "
                  "transactions, including bookkeeping income statements balance "
                  "sheets and tax accounting",
    "dividend": "a dividend is a cash payment a company distributes to its "
                "shareholders out of profits, paid per share on stocks and held "
                "in brokerage accounts",
    "mortgage": "a mortgage is a loan to buy real estate property, secured by the "
                "home, repaid with interest in monthly payments over years",
    "roth": "a Roth IRA is a retirement investment account funded with after tax "
            "dollars where qualified withdrawals including gains are tax free",
    "401k": "a 401k is an employer sponsored retirement savings plan where pre tax "
            "salary is invested in funds, often with an employer match, taxed on "
            "withdrawal",
    "capital": "capital gains are the profit from selling an investment such as "
               "stocks or real estate for more than its purchase price, subject to "
               "capital gains tax",
    "interest": "interest is the cost of borrowing money or the return on savings, "
                "charged on loans and credit and paid on deposits and bonds",
}
