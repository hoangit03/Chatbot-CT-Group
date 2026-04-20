from paddlex import create_pipeline
p = create_pipeline('PP-StructureV3')
res = p('/data/input/report.pdf')
for x in res:
    print(x)
