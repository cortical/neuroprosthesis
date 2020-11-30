%the spinal cord section is divided in local regions of interest.
%feat_filter (section_height x section_width x n_feat) define what region
%   of the section is captured by each feature.

%featmatrix is a n_rats x n_feat array describing the presence of spared 
%   tissue within a local region of interest. 
nfeat=size(featmatrix,2);

%mask_evaluated (section_height x section_width) indicated the are of the 
%   spinal cord which is considered for the present evaluation. All other 
%   areas were either spared on no animals, or spared on all animals.

%P7_ladder and P28_ladder are vectors reporting the performance in ladder
%   crossing, for each rat, 7 and 28 days after injury, respectively.
%Data are in % points, left leg.
%Subject 1 to 6 of day 7 were later trained using neurostimulation. Since
%   they did not undergo spontaneous recovery, they were excluded from this
%   analysis at 28 days post injury (spontaneous recovery only).

H=2; %hidden neurons
m=1; %output neurons
mu=0.4; %Learning-rate parameter
alpha=0.02; %momentum parameter
epochMax = 100000; %max number of training epochs (ending criterion)
MSETarget= 0.0005; %target min squared error (alternate ending criterion)


%% Fig S6B left

%training multi-layer perceptron (MLP)
[Wx,Wy]=E_trainMLP(nfeat,H,m,mu,alpha,featmatrix',P7_ladder(:,1)',epochMax,MSETarget);
%extracting response to unit inputs
Y1=E_runMLP(eye(nfeat),Wx,Wy);

b=Y1-min(Y1); %normalization
%recomposing a two-dimensional image using each feature unit responses
showreg=zeros(size(mask_evaluated));
for i=1:nfeat
    showreg=showreg+feat_filter(:,:,i).*mask_evaluated*b(i);
end

%visualizing result
figure
Iblur=blurpad(showreg,12); %smoothing image for visualization
%normalization
Iblur=Iblur-min(min(Iblur.*mask_evaluated+ (Iblur.*mask_evaluated==0)));
Iblur=Iblur./max(max(Iblur));
imagesc(Iblur.*mask_evaluated) %visualization



%% Fig S6B right

n_rats_nonspont=6; %the first 6 rats did not recover spontaneously (neurostimulation)

featmatrixshort=featmatrix(n_rats_nonspont+1:end,:); %reduced feature matrix
%training multi-layer perceptron (MLP)
[Wx,Wy]=E_trainMLP(size(featmatrixshort,2),2,1,0.4,0.02,featmatrixshort',P28_ladder(:,1)',100000,0.0005);
%extracting response to unit inputs
Y1=E_runMLP(eye(size(featmatrixshort,2)),Wx,Wy);

b=Y1-min(Y1); %normalization
%recomposing a two-dimensional image using each feature unit responses
showreg=zeros(size(mask_evaluated));
for i=1:nfeat
    showreg=showreg+feat_filter(:,:,i).*mask_evaluated*b(i);    
end

%visualizing result
figure
Iblur=blurpad(showreg,12); %smoothing image for visualization
%normalization
Iblur=Iblur-min(min(Iblur.*mask_evaluated+ (Iblur.*mask_evaluated==0)));
Iblur=Iblur./max(max(Iblur));
imagesc(Iblur.*mask_evaluated) %visualization




%% MLP implementation

function [Wx,Wy,MSE]=E_trainMLP(p,H,m,mu,alpha,X,D,epochMax,MSETarget)
% The matrix implementation of the Backpropagation algorithm for two-layer
% Multilayer Perceptron (MLP) neural networks.
%
% Author: Marcelo Augusto Costa Fernandes
% DCA - CT - UFRN
% mfernandes@dca.ufrn.br
%
% Note: the authors of the present article have corrected some inaccuracy
% in the original implementation.
%
% Input parameters:
%   p: Number of the inputs.
%   H: Number of hidden neurons
%   m: Number of output neurons
%   mu: Learning-rate parameter
%   alpha: Momentum constant
%   X: Input matrix.  X is a (p x N) dimensional matrix, where p is a number of the inputs and N is a training size.
%   D: Desired response matrix. D is a (m x N) dimensional matrix, where m is a number of the output neurons and N is a training size.
%   epochMax: Maximum number of epochs to train.
%   MSETarget: Mean square error target.
%
% Output parameters:
%   Wx: Hidden layer weight matrix. Wx is a (H x p+1) dimensional matrix.
%   Wy: Output layer weight matrix. Wy is a (m x H+1) dimensional matrix.
%   MSE: Mean square error vector. 

[p1 N] = size(X);
bias = -1;
X = [bias*ones(1,N) ; X];
Wx = rand(H,p+1)./(p+1); %MB: corrected
WxAnt = zeros(H,p+1);
Tx = zeros(H,p+1);
Wy = rand(m,H+1)./(H+1); %MB: corrected
Ty = zeros(m,H+1);
WyAnt = zeros(m,H+1);
dWyAnt=zeros(m,H+1); %MB: corrected
dWxAnt=zeros(H,p+1); %MB: corrected
DWy = zeros(m,H+1);
DWx = zeros(H,p+1);
MSETemp = zeros(1,epochMax);
for i=1:epochMax    
    k = randperm(N);
    X = X(:,k);
    D = D(:,k);
    V = Wx*X;
    Z = 1./(1+exp(-V));
    S = [bias*ones(1,N);Z];
    G = Wy*S;
    Y = 1./(1+exp(-G));
    E = D - Y;
    mse = mean(mean(E.^2));
    MSETemp(i) = mse;
    % disp(['epoch = ' num2str(i) ' mse = ' num2str(mse)]);
    if (mse < MSETarget)
        MSE = MSETemp(1:i);
        return
    end 
    df = Y.*(1-Y);
    dGy = df .* E;
    DWy = mu/N * dGy*S';
    Ty = Wy;
    Wy = Wy + DWy + alpha*dWyAnt; %MB: corrected
    WyAnt = Wy-Ty; %MB: corrected
    df= S.*(1-S);
    dGx = df .* (Wy' * dGy);
    dGx = dGx(2:end,:);
    DWx = mu/N* dGx*X';
    Tx = Wx;
    Wx = Wx + DWx + alpha*dWxAnt; %MB: corrected
    WxAnt = Wx-Tx; %MB: corrected
end
MSE = MSETemp;
end


function Y=runMLP(X,Wx,Wy)
% The matrix implementation of the two-layer Multilayer Perceptron (MLP) neural networks.
%
% Author: Marcelo Augusto Costa Fernandes
% DCA - CT - UFRN
% mfernandes@dca.ufrn.br
%
% Input parameters:
%   X: Input neural network.  X is a (p x K) dimensional matrix, where p is a number of the inputs and K >= 1.
%   Wx: Hidden layer weight matrix. Wx is a (H x p+1) dimensional matrix.
%   Wy: Output layer weight matrix. Wy is a (m x H+1) dimensional matrix.
%
% Output parameters:
%  Y: Outpuy neural network.  Y is a (m x K) dimensional matrix, where m is a number of the output neurons and K >= 1.

[p1 N] = size (X);
bias = -1;
X = [bias*ones(1,N) ; X];
V = Wx*X;
Z = 1./(1+exp(-V));
S = [bias*ones(1,N);Z];
G = Wy*S;
Y = 1./(1+exp(-G));

end