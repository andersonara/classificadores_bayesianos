library(readr)

dados_brutos <- read_tsv("amazon_alexa.tsv")

Y <- ifelse(dados_brutos$rating<5,'1','0')


library(tm)

corpus <- Corpus(VectorSource(dados_brutos$verified_reviews))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords())
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)

specialChars <- function(x) gsub("[^\x01-\x7F]","", x)
corpus <- tm_map(corpus, 
                 content_transformer(specialChars))

inspect(corpus[1:15])

DTM <- DocumentTermMatrix(corpus)
DTM <- removeSparseTerms(DTM, 0.95) 
findFreqTerms(DTM, 300)





DTM2 <- as.matrix(DTM)
DTM2[DTM2>=2]=1
#View(DTM)
dados <- data.frame(Y,apply(DTM2,2,as.factor),stringsAsFactors = T)
#dados <- data.frame(Y,apply(DTM2,2,as.numeric))

ncol(dados)

str(dados)
#dados2 <- dados[1:1000,1:10]

require(caret)
set.seed(1807)
trein.index <- createDataPartition(dados$Y, p=0.75, list=FALSE)
dados.trein <- dados[ trein.index,]
dados.teste <- dados[-trein.index,]

#NB
require(e1071)
mod.nb <- naiveBayes(Y~.,data=dados.trein)
p.nb <- predict(mod.nb,newdata = dados.teste,
                type='raw')

boxplot(p.nb[,2]~dados.teste$Y)
Y.nb<- ifelse(p.nb[,2]>0.5,1,0)
MC.NB <- table(dados.teste$Y,Y.nb)

sum(diag(MC.NB))/sum(MC.NB)

cor(as.numeric(Y.nb),as.numeric(dados.teste$Y))

require(bnclassify)

mod.tan <- tan_cl('Y', dados.trein)
mod.tan <- lp(mod.tan, dados.trein, smooth = 1)
p.tan<- predict(mod.tan, dados.teste, prob = TRUE)
boxplot(p.tan[,2]~dados.teste$Y)
Y.tan<- ifelse(p.tan[,2]>0.5,1,0)
MC.TAN <- table(dados.teste$Y,Y.tan)



mod.kdb <- kdb('Y', k=3, kdbk = 2,dados.trein)
mod.kdb <- lp(mod.kdb, dados.trein, smooth = 1)
p.kdb<- predict(mod.kdb, dados.teste, prob = TRUE)
boxplot(p.kdb[,2]~dados.teste$Y)
Y.kdb<- ifelse(p.kdb[,2]>0.5,1,0)
MC.KDB <- table(dados.teste$Y,Y.kdb)



mod.aode <- aode('Y', dados.trein)
mod.aode <- lp(mod.aode, dados.trein, smooth = 1)
p.aode <- predict(mod.aode, dados.teste, prob = TRUE)
boxplot(p.aode[,2]~dados.teste$Y)
Y.aode<- ifelse(p.aode[,2]>0.5,1,0)
MC.AODE <- table(dados.teste$Y,Y.aode)



#RL
mod.rl <- glm(Y~.,data=dados.trein,
              family = binomial(link='logit'))
p.rl <- predict(mod.rl,newdata = dados.teste,
                type='response')

boxplot(p.rl~dados.teste$Y)
Y.rl<- ifelse(p.rl>0.5,1,0)
MC.RL <- table(dados.teste$Y,Y.rl)


MCC <- function(MC){
  TP=MC[2,2]; TN=MC[1,1]; FP=MC[1,2]; FN=MC[2,1]
  A=(TP*TN-FP*FN)
  B=(sqrt(TP+FP)*sqrt(TP+FN)*sqrt(TN+FP)*sqrt(TN+FN))
  return(A/B)
}

F1 <- function(MC){
  TP=MC[2,2]; TN=MC[1,1]; FP=MC[1,2]; FN=MC[2,1]
  SEN=TP/(FN+TP)
  PRE=TP/(TP+FP)
  F1=2*SEN*PRE/(SEN+PRE)
  return(F1)
}

#MCCs
MCC(MC.NB)
MCC(MC.TAN)
MCC(MC.KDB)
MCC(MC.AODE)
MCC(MC.RL)

#F1
F1(MC.NB)
F1(MC.TAN)
F1(MC.KDB)
F1(MC.AODE)
F1(MC.RL)




#PLOT MATRIZ DE CONFUSÃO


names(dimnames(MC.NB)) <- c('real','predito')

fourfoldplot(MC.NB, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "NB")


names(dimnames(MC.TAN)) <- c('real','predito')

fourfoldplot(MC.TAN, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "TAN")


names(dimnames(MC.KDB)) <- c('real','predito')

fourfoldplot(MC.KDB, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "KDB")

names(dimnames(MC.AODE)) <- c('real','predito')

fourfoldplot(MC.AODE, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "AODE")


names(dimnames(MC.RL)) <- c('real','predito')

fourfoldplot(MC.RL, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "RL")


### CURVA ROC 

roc <- function(y,out){
  tau <- sort(out, index.return=TRUE)
  n   <- length(tau$x);
  s   <- c()
  e   <- c()
  for(c in 1:n){
    aux = as.numeric(out >= tau$x[c]);
    s[c] = sum(y*aux)/sum(y);
    e[c] = sum(as.numeric(!y)*as.numeric(!aux))/sum(as.numeric(!y));
  }
  return( list(s=s,e=e,tau=tau$x) )
}

Y.tre <- as.numeric(dados.trein$Y)-1




#NB
p.nbT <- predict(mod.nb, newdata = dados.trein,
                type='raw')
eroc=roc(Y.tre,p.nbT[,2])
pares=cbind(1-eroc$e,eroc$s)


distan=numeric(0)
for (i in 1:length(Y.tre)) {
  
  distan[i]=sqrt((pares[i,1]-0)^2+(pares[i,2]-1)^2)
}

ordem=cbind(1:length(Y.tre),distan)


##### CALCULO DO PONTO DE CORTE #####
o=min(ordem[ordem[,2]==min(distan),1])
corte=eroc$tau
Ct.nb=mean(corte[o])
if(is.na(Ct.nb)) Ct=0.5



#TAN
p.tanT <- predict(mod.tan, dados.trein,
                 prob=T)
eroc=roc(Y.tre,p.tanT[,2])
pares=cbind(1-eroc$e,eroc$s)


distan=numeric(0)
for (i in 1:length(Y.tre)) {
  
  distan[i]=sqrt((pares[i,1]-0)^2+(pares[i,2]-1)^2)
}

ordem=cbind(1:length(Y.tre),distan)


##### CALCULO DO PONTO DE CORTE #####
o=min(ordem[ordem[,2]==min(distan),1])
corte=eroc$tau
Ct.tan=mean(corte[o])
if(is.na(Ct.tan)) Ct.kdb=0.5


#KDB
p.kdbT <- predict(mod.kdb, dados.trein,
                prob=T)
eroc=roc(Y.tre,p.kdbT[,2])
pares=cbind(1-eroc$e,eroc$s)


distan=numeric(0)
for (i in 1:length(Y.tre)) {
  
  distan[i]=sqrt((pares[i,1]-0)^2+(pares[i,2]-1)^2)
}

ordem=cbind(1:length(Y.tre),distan)


##### CALCULO DO PONTO DE CORTE #####
o=min(ordem[ordem[,2]==min(distan),1])
corte=eroc$tau
Ct.kdb=mean(corte[o])
if(is.na(Ct.kdb)) Ct.kdb=0.5




#KDB
p.aodeT<- predict(mod.aode, dados.trein,
                 prob=T)
eroc=roc(Y.tre,p.aodeT[,2])
pares=cbind(1-eroc$e,eroc$s)


distan=numeric(0)
for (i in 1:length(Y.tre)) {
  
  distan[i]=sqrt((pares[i,1]-0)^2+(pares[i,2]-1)^2)
}

ordem=cbind(1:length(Y.tre),distan)


##### CALCULO DO PONTO DE CORTE #####
o=min(ordem[ordem[,2]==min(distan),1])
corte=eroc$tau
Ct.aode=mean(corte[o])
if(is.na(Ct.aode)) Ct.aode=0.5




#RL
p.rlT <- predict(mod.rl,newdata = dados.trein,
        type='response')

eroc=roc(Y.tre,p.rlT)
pares=cbind(1-eroc$e,eroc$s)


distan=numeric(0)
for (i in 1:length(Y.tre)) {
  
  distan[i]=sqrt((pares[i,1]-0)^2+(pares[i,2]-1)^2)
}

ordem=cbind(1:length(Y.tre),distan)


##### CALCULO DO PONTO DE CORTE #####
o=min(ordem[ordem[,2]==min(distan),1])
corte=eroc$tau
Ct.rl=mean(corte[o])
if(is.na(Ct.rl)) Ct.rl=0.5


#NOVOS PONTOS DE CORTE
data.frame(Ct.nb,Ct.tan,Ct.kdb,Ct.aode,Ct.rl)

Y.nb<- ifelse(p.nb[,2]>Ct.nb,1,0)
MC.NB <- table(dados.teste$Y,Y.nb)

Y.tan<- ifelse(p.tan[,2]>Ct.tan,1,0)
MC.TAN <- table(dados.teste$Y,Y.tan)

Y.kdb<- ifelse(p.kdb[,2]>Ct.kdb,1,0)
MC.KDB <- table(dados.teste$Y,Y.kdb)

Y.aode<- ifelse(p.aode[,2]>Ct.aode,1,0)
MC.AODE <- table(dados.teste$Y,Y.aode)

Y.rl<- ifelse(p.rl>Ct.rl,1,0)
MC.RL <- table(dados.teste$Y,Y.rl)




#PLOT MATRIZ DE CONFUSÃO


names(dimnames(MC.NB)) <- c('real','predito')

fourfoldplot(MC.NB, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "NB")


names(dimnames(MC.TAN)) <- c('real','predito')

fourfoldplot(MC.TAN, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "TAN")


names(dimnames(MC.KDB)) <- c('real','predito')

fourfoldplot(MC.KDB, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "KDB")

names(dimnames(MC.AODE)) <- c('real','predito')

fourfoldplot(MC.AODE, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "AODE")


names(dimnames(MC.RL)) <- c('real','predito')

fourfoldplot(MC.RL, color = c("gray", "tomato"),
             conf.level = 0, margin = 1, main = "RL")


#MCCs
MCC(MC.NB)
MCC(MC.TAN)
MCC(MC.KDB)
MCC(MC.AODE)
MCC(MC.RL)

#F1
F1(MC.NB)
F1(MC.TAN)
F1(MC.KDB)
F1(MC.AODE)
F1(MC.RL)
