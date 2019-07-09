def savetext(filename,contents):
    contents=str(contents)
    fh = open(filename, 'a', encoding='utf-8')
    fh.write(contents + "\n")
    fh.close()