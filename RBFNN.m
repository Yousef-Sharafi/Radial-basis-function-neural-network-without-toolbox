clc;
close all;
clear all;

n=50;
m=3000;
x1=zeros(n,1);
y1=zeros(n,1);

maxEpoch=1000;
eta=0.001;

numberOfNeuron=30;
LowerBound=-1;
UperBound=1;

r1=1.3;
for i=1:n
    if rand<0.7
      r=r1;
    else
      r=unifrnd(0,r1); 
    end
    teta=unifrnd(0,2*pi);
    x1(i)=r*sin(teta);
    y1(i)=r*cos(teta);
end

d1=[x1 y1 ones(n,1)];
x2=zeros(n,1);
y2=zeros(n,1);

for i=1:n
    if rand<0.7
      r=r1;
    else
      r=unifrnd(0,r1); 
    end
    teta=unifrnd(0,2*pi);
    x2(i)=r*sin(teta)-1.5;
    y2(i)=r*cos(teta)-2;
end

d2=[x2 y2 2*ones(n,1)];
x3=zeros(n,1);
y3=zeros(n,1);

for i=1:n
    if rand<0.7
      r=r1;
    else
      r=unifrnd(0,r1); 
    end
    teta=unifrnd(0,2*pi);
    x3(i)=r*sin(teta)-3;
    y3(i)=r*cos(teta);
end

d3=[x3 y3 3*ones(n,1)];
Sigma=[1 1];
Num=n;

f1 = mvnrnd([0 5], Sigma, Num); 
f2 = mvnrnd([5 0], Sigma, Num);
f3 = mvnrnd([8 8], Sigma, Num);

data=[d1
    d2
    d3];

for p=1:2
    min1=min(data(:,p));
    max1=max(data(:,p));
    data(:,p)=(data(:,p)-min1)/(max1-min1);
end

plot(data(1:n,1),data(1:n,2),'o',...
    'MarkerSize',9,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','b')
hold on;
plot(data(n+1:2*n,1),data(n+1:2*n,2),'o',...
    'MarkerSize',9,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','r')

hold on;
plot(data(2*n+1:3*n,1),data(2*n+1:3*n,2),'o',...
    'MarkerSize',9,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','g')

[a,b]=kmeans(data(:,1:2),numberOfNeuron);
sigma=zeros(1,numberOfNeuron);

for k=1:numberOfNeuron
    sum1=0;
    dataTemp=data(find(a==k),:);
    for i=1:size(dataTemp,1)
        sum1=sum1+norm(dataTemp(i,1:2)-b(k,:));
    end
    sigma(k)=sum1/size(dataTemp,1);
end

sigma=0.1*ones(1,numberOfNeuron);
indexZero=find(sigma==0);
indexNotZero=find(sigma~=0);
sigma(indexZero)=min(sigma(indexNotZero));
w1=unifrnd(LowerBound,UperBound,[1 numberOfNeuron]);
bais=rand;

for iter=1:maxEpoch  
    index=randperm(3*n);
    data=data(index,:); 
    for j=1:3*n
        input=data(j,1:2);
        target=data(j,3);
        output1=zeros(numberOfNeuron,1);
        
        for k=1:numberOfNeuron
            sig=(2*(sigma(k)^2));
            output1(k)=exp((-1/sig)*(norm(input-b(k,:))^2));
        end
        
        net1=w1*output1;
        output2=net1+bais;
        error=target-output2;
      
        w1=w1-eta*error*-1*1*output1';
        bais=bais-eta*error*-1*1;
    end   
end

outputRBF=zeros(1,3*n);

for j=1:3*n
    input=data(j,1:2);
    target=data(j,3);
    output1=zeros(numberOfNeuron,1);
    for k=1:numberOfNeuron
        sig=(2*(sigma(k)^2));
        output1(k)=exp((-1/sig)*(norm(input-b(k,:))^2));
    end
    net1=w1*output1;
    output2=net1+bais;
    outputRBF(j)=round(output2);
end

dataNN=[data outputRBF'];
count=0;

for p=1:3*n
    if(dataNN(p,3)==dataNN(p,4))
        count=count+1;
    end
end

(100*count)/(3.0*n)

dataTX=unifrnd(0,1,[1 m]);
dataTY=unifrnd(0,1,[1 m]);

hold on;

plot(dataTX,dataTY,'*',...
    'MarkerSize',7,...
    'MarkerEdgeColor','black',...
    'MarkerFaceColor','black');

dataTestN=[dataTX' dataTY' zeros(m,1)];

for j=1:m
    input=dataTestN(j,1:2);
    target=dataTestN(j,3);
    output1=zeros(numberOfNeuron,1);
    for k=1:numberOfNeuron
        sig=(2*(sigma(k)^2));
        output1(k)=exp((-1/sig)*(norm(input-b(k,:))^2));
    end
    net1=w1*output1;
    output2=net1+bais;
    dataTestN(j,3)=round(output2);
end


index1=find(dataTestN(:,3)==1);
index2=find(dataTestN(:,3)==2);
index3=find(dataTestN(:,3)==3);

hold on;

plot(dataTestN(index1,1),dataTestN(index1,2),'*',...
    'MarkerSize',7,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor','b');

hold on;

plot(dataTestN(index2,1),dataTestN(index2,2),'*',...
    'MarkerSize',7,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r');

hold on;

plot(dataTestN(index3,1),dataTestN(index3,2),'*',...
    'MarkerSize',7,...
    'MarkerEdgeColor','g',...
    'MarkerFaceColor','g');

hold on;
plot(b(:,1),b(:,2),'p',...
    'MarkerSize',20,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','k')


