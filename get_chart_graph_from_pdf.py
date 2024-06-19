

tabs = page.find_tables()  # locate and extract any tables on page
if tabs.tables:  # at least one table found?
    for tabno, tab in enumerate(tabs, start=1):
        table_extract = tab.extract()  # get the table extracted from the page
        # print(table_extract)  # print content of first table
        pdfPageDescriptor.tables.append(table_extract)