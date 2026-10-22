import json, math, statistics as st
B=r"C:\Users\emari\Desktop\MIA IA LOCALE\.claude\worktrees\robotica-congela-previsione-legge-1-929f74\strumenti\termometro\risultati"
runs=[json.load(open(B+r"\q15_kaggle_B\q15\run_s%d.json"%s)) for s in range(101,113)]
at=lambda r,s:[x for x in r if x["step"]==s][0]
D=[at(r,40)["D_seq"] for r in runs]; E=[at(r,40)["ess"] for r in runs]
def corr(a,b):
    ma,mb=st.mean(a),st.mean(b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
print("corr(log ESS40, D_seq40)=",corr([math.log(e) for e in E],D))
print("corr(log ESS40, |D_seq40-0.485|)=",corr([math.log(e) for e in E],[abs(x-0.485) for x in D]))
order=sorted(range(12),key=lambda i:E[i])
print([(101+i,round(E[i],1),round(D[i],3)) for i in order])
# frazione delle corse con D_seq40 sopra / sotto 0.485
print("sopra 0.485:",sum(x>0.485 for x in D),"sotto:",sum(x<0.485 for x in D), "media", st.mean(D))
# quanto D_seq si muove in tutto: |D(40)-D(0)| medio
print("media |D40-D0|:",st.mean([abs(x-0.485) for x in D]))
# SD D_seq stimata se ESS fosse binomiale-like? descrittivo: SD attesa di un rapporto di Bernoulli con ESS n_eff, p=.485
print("SD binomiale con ESS mediana 15.1 e p=0.485:", math.sqrt(0.485*0.515/15.1))
