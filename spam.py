from math import log
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

file= open("SMSSpamCollection","r") #abre o ficheiro com os emails


palavras=[] #ira guardar todas as palavras dos emails
dicionario=[] #ira conter as palavras dos emails sem repeticoes
m_ham=0 #ira conter o nr de emails ham
m_spam=0 #ira conter o nr de emails spam
p_ham=[] #ira guardar as probabilidades de cada palavra para que o email seja ham
p_spam=[] #ira guardar as probabilidades de cada palavra para que o email seja spam


x_default=[] #serve para reniciar o array com as frequencias das palavras

X=[] #ira conter os nrs dos emails (ids do emails)
y=[] #label de cada email 

mails=[] #ira guardar os emails 
id_mail=0


nr_aleatorio=random.randint(2,50) #gera um nr aleatorio para que a divisao dos dados de treino teste e validacao seja smpre diferente

#Criacao do dataset
for line in file:
            
    #se o mail for spam o valor da label é 1 se ham o valor da label é -1
    if line.split(None,1)[0]=="spam":
        y.append(1)
    else:
        y.append(-1)  
        
    #guarda o texto de cada email
    mails.append(line.split(None,1)[1])
    
    #guarda o nr do email
    X.append(id_mail)
    
    #actualiza o id para a proximo email
    id_mail=id_mail+1
    


#Divisao do dataset em train(60%) validation(20%) e test set(20%)
    
#Divisao do test set (20%) e train set temporario (80%)
X_train_tmp, X_test, y_train_tmp, y_test = train_test_split(X, y, test_size=0.2,train_size=0.8,random_state=nr_aleatorio)

#Divisao dos 80% do train set temporario no train set real e validation set em que test size corresponde ao validation size
X_train, X_val, y_train, y_val = train_test_split(X_train_tmp, y_train_tmp, test_size=0.25, train_size =0.75,random_state=nr_aleatorio)     









#Treino do modelo usando o Training set

for i in X_train:
    
    

    #Verifica quantos emails ha de spam e ham no training set
    if y[i]==-1:
        m_ham=m_ham+1
    if y[i]==1:
        m_spam=m_spam+1    
    
 
    #Divide a mensagem de cada email em palavras e percorre cada palavra
    for word in mails[i].split(" "):
        
        
        #adiciona todas as palavras ao array de palavras que ira conter as palavras de todos os emails
        if len(word)>=3: #tamanho de cada palavra tem que ser superior a 3 caracteres
            palavras.append(word)
            


#retira as palavras duplicadas que existem no array de palavras e cria o array dicionario que contem palavras nao duplicadas existentes no ficheiro dos emails
dicionario=list(set(palavras))



 
#inicializa os arrays que vao contar as frequencias de cada palavra a 1 para o spam e ham em que o tamanho total de cada array corresponde ao nr total de palavras que estao no dicionario     
for i in range (len(dicionario)):
   p_ham.append(1)
   p_spam.append(1)
   
   x_default.append(0) #inicializa o array que ira servir para reniciar o array x[] que contem as frequencias das palavras dos emails para os dados de teste






#Adiciona as Frequencias para os arrays p_spam e p_ham      
for j in range (len(dicionario)): 
    
    
    for i in X_train:
        
        #Define o indicador se o email é ham ou spam
        if y[i]==1:
            index=0
        if y[i]==-1:
            index=1
            
        
        for word in mails[i].split(" "):
            
            #Se encontrar a palavra do dicionario no respectivo email entao incrementa um valor no respectivo array e posição consoante o index
            if word == dicionario[j]:
                
                
                if index==0:
                    p_spam[j]=p_spam[j]+1
 
                    
                if index==1:
                    p_ham[j]=p_ham[j]+1                    
 



#Transforma as frequencias de cada palavra em probabilidades freq_da_palavra/nr_total_emails(spam ou ham)                   
for i in range (len(p_spam)):
    p_spam[i]=p_spam[i]/len(p_spam)
    p_ham[i]=p_ham[i]/len(p_ham)    

        
            
m=m_spam+m_ham #nr total de emails (spam+ham)       

print("spam",m_spam)        
print("ham",m_ham)        
print("total",m)





#inicializa o array de frequencias para os dados de treino que ira ter as frequencias de palavras para cada email nos dados de treino
X_train_freq = [[0]*len(dicionario)]*len(X_train)
#inicializa o array de frequencias para os dados de teste que ira ter as frequencias de palavras para cada email nos dados de teste
X_test_freq = [[0]*len(dicionario)]*len(X_test)
#inicializa o array de frequencias para os dados de validacao que ira ter as frequencias de palavras para cada email nos dados de validacao
X_val_freq = [[0]*len(dicionario)]*len(X_val)




id_mail=0
#preenchimento do array com as frequencias das palavras de cada email dos dados de teste
for i in X_test:
    
    mail_words=[]
    
    
    X_test_freq[id_mail]=copy.copy(x_default) # é necessario inializar a 0 de novo para cada posiçao pq de alguma maneira esta preencher a posição a seguir com valores
    
    
    for word in mails[i].split(" "): 
        if len(word)>=3:
            mail_words.append(word)


    for j in range (len(dicionario)):
    
              
        if dicionario[j] in mail_words:
            contador=mail_words.count(dicionario[j])
            #print("passa aqui mail",i,"palavra",j)
            
            X_test_freq[id_mail][j]=X_test_freq[id_mail][j]+contador  
            
    id_mail=id_mail+1           




id_mail=0
#preenchimento  do array com as frequencias das palavras de cada email dos dados de treino
for i in X_train: 
    
    mail_words=[]
    X_train_freq[id_mail]=copy.copy(x_default) # é necessario inializar a 0 de novo para cada posiçao pq de alguma maneira esta preencher a posição a seguir com valores
    
    
    #converte a mensagem para palavra a palavra
    for word in mails[i].split(" "): 
        if len(word)>=3:
            mail_words.append(word)


    for j in range (len(dicionario)):
    
             
        if dicionario[j] in mail_words:
            contador=mail_words.count(dicionario[j])

            X_train_freq[id_mail][j]=X_train_freq[id_mail][j]+contador  

    id_mail=id_mail+1
    
    
    
id_mail=0
#preenchimento  do array com as frequencias das palavras de cada email dos dados de validacao
for i in X_val: 
    
    mail_words=[]
    X_val_freq[id_mail]=copy.copy(x_default) # é necessario inializar a 0 de novo para cada posiçao pq de alguma maneira esta preencher a posição a seguir com valores
    
    
    #converte a mensagem para palavra a palavra
    for word in mails[i].split(" "): 
        if len(word)>=3:
            mail_words.append(word)


    for j in range (len(dicionario)):
    
             
        if dicionario[j] in mail_words:
            contador=mail_words.count(dicionario[j])

            X_val_freq[id_mail][j]=X_val_freq[id_mail][j]+contador  

    id_mail=id_mail+1    



# MODELO NAIVE BAYES 

#valores que c ira tomar
valores=[0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,1]
percent=[]
for c in valores:

    #inicializa b
    b=log(c)+log(m_ham)-log(m_spam)
    
    y_pred=[] #inicializa o array que ira guardar as previsoes das labels calculadas pelo algoritmo
    
    id_mail=0
    
    
    for i in X_test: 
        
        #inicializa t
        t=-b
        
        
    
        for j in range (len(dicionario)):
            
            #actualiza o valor de t para o mail em questao                    
            t=t+(X_test_freq[id_mail][j]*(log(p_spam[j])-log(p_ham[j]))) 
            
            
        #se o t para o mail em questao for >0 entao esse mail é spam senao é ham    
        if t>0:
            y_pred.append(1)    #spam
        else:    
            y_pred.append(-1)    #ham        
                
        id_mail=id_mail+1        
        
   
    
    erros=0
    
    
    contador=0
            
    #Calculo da accuracy
    for i in X_test:
        
        #Verifica se a label prevista pelo algoritmo é igual ao output real
        if y[i]!=y_pred[contador]:
            erros=erros+1
            
    
        contador=contador+1
    
    print("Accuracy Naive Bayes para c=",c,(len(X_test)-erros)/len(X_test))
    
    percent.append((len(X_test)-erros)/len(X_test))

plt.plot(valores,percent)
plt.xlabel("Valores de C")
plt.ylabel("Accuracy")   
plt.show() 

# MODELO SVM 
    
    
    
# Pré Tratamento dos dados


X_train_freq=np.array(X_train_freq)
y_train=np.array(y_train)
y_train=y_train.reshape(-1,1)
y_train=y_train.ravel()



X_test_freq=np.array(X_test_freq)
y_test=np.array(y_test)
y_test=y_test.reshape(-1,1)
y_test=y_test.ravel()


X_val_freq=np.array(X_val_freq)
y_val=np.array(y_val)
y_val=y_val.reshape(-1,1)
y_val=y_val.ravel()

percent=[]

for c in valores:
#Inicializacao do modelo no modo linear
    model = SVC( kernel='linear',C=c)
    
    
    clf = model.fit(X_train_freq, y_train)
    
    #Calculo da accuracy
    pred = clf.predict(X_val_freq)
    score_train = model.score(X_train_freq, y_train)
    score_test = model.score(X_test_freq, y_test)
    
    print("para c=",c)
    print("Accuracy Train SVM:", score_train)
    print("Accuracy Validation SVM:",(np.mean(y_val == pred)))        
    print("Accuracy Test SVM:", score_test)
    print("-------------------------")

    percent.append(score_test)

plt.plot(valores,percent)
plt.xlabel("Valores de C")
plt.ylabel("Accuracy")   
plt.show() 


# Algoritmo Preceptao

#inicializa a bias
b=0

#Nr de epocas
max_iter=40

#Inicializa o vector pesos em que o seu tamanho corresponde ao nr de features (palavras) da matriz x para que se possa fazer o produto interno
w=[0]*len(dicionario)
w=np.array(w)

error_record=[]
iter_record=[]

#Preceptao Training 
for e in range (max_iter):
    
    erros=0
    for i in range (len(X_train)):
        #Calculo da activacao (produto interno do vector w pela matriz de palavras do mail em questao mais a bias)
        a=np.dot(w,X_train_freq[i]) + b


        #se o resultado da activacao for o oposto da label real ou seja se der um valor negativo entao:
        if y[X_train[i]]*a<=0:
            erros=erros+1
            #actualizase a bias 
            b=b+y[X_train[i]]
            for j in range (len(dicionario)):
                #actualiza-se os pesos em cada posicao com o objectivo ter um w ideal para que nao de erro na proxima iteracao
                w[j]=w[j] + y[X_train[i]] * X_train_freq[i][j] # a cada posicao de w(peso da jésima palavra) é somado o produto do valor da label real(1 ou -1) do email em questao pela freq da jésima palavra do email em questao
            
    error_record.append(erros)            
    iter_record.append(e)
                

plt.plot(iter_record,error_record)
plt.xlabel("Epocas")
plt.ylabel("Erros")   
plt.show() 



a=0
erros=0

#Preceptao Test
for i in range (len(X_test)):
    #calculo da activacao para os dados de teste
    a=np.dot(w,X_test_freq[i]) + b
    
    #verifica se preveu bem ou mal e calcula a acuracy
    if y[X_test[i]]*a<0:
        erros=erros+1

    

print("Accuracy Preceptao",(len(X_test)-erros)/len(X_test))   