import matplotlib.pyplot as plt

score = open("logs_atari_entraine2/score.txt", "r")
partie = [0]
result = [0]
p = 0
r = 0
c = 0
for l in score :
    s = l.split(',')
    r += int(s[1].split('\n')[0])
    c+=1
    if c ==50 :
        partie.append(int(s[0]))
        result.append(r/50.)
        r=0
        c=0
    
score.close()
plt.plot(partie, result)
plt.show()