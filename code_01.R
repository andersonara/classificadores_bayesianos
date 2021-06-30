### Gerando amostra ----------------------------------------------
n=1000

set.seed(1807)
X1.0 <- rnorm(n/2,mean=3,sd=1)
X2.0 <- rnorm(n/2,mean=3,sd=1)

X1.1 <- rnorm(n/2,mean=1,sd=1)
X2.1 <- rnorm(n/2,mean=1,sd=1)

Y <- c(rep(0,n/2),rep(1,n/2))
X1 <- c(X1.0,X1.1)
X2 <- c(X2.0,X2.1)

dados <- data.frame(Y,X1,X2)

### Visualizando amostra ----------------------------------------------
par(mfrow=c(1,2))
plot(X1,X2,col='gray')
plot(X1,X2,col=Y+2)

cor(X1,X2)



### Gerando grid em X1 e X2 -------------------------------------------
require(dplyr)
Min <- dados %>%  select(-Y) %>% summarise_all(min)
Max <- dados %>% select(-Y) %>% summarise_all(max)

p=ncol(dados)-1
X=matrix(0,ncol=p,nrow=50)
for (i in 1:p){
  X[,i]=seq(as.numeric(Min[i]),
            as.numeric(Max[i]),
            length.out = 50)
  
} 

XX=expand.grid(X[,1],X[,2])

names(XX) <- c('X1','X2')

plot(X1,X2,col=Y+2)
points(XX,pch='.')

### Ajuste naive bayes -------------------------------------------------
require(e1071)
mod.nb <- naiveBayes(Y~.,data=dados)
p.nb <- predict(mod.nb,newdata = XX,
                type='raw')
Y.nb <- ifelse(p.nb[,2]>0.5,1,0)
points(XX,col=Y.nb+2)


### Ajuste regressão logística  ----------------------------------------
mod.rl <- glm(Y~.,data=dados,family = binomial)
p.rl <- predict(mod.rl,newdata = XX,
                type='response')
Y.rl<- ifelse(p.rl>0.5,1,0)
plot(X1,X2,col=Y+2)
points(XX,col=Y.rl+2)
