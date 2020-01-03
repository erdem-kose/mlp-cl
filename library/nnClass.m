classdef nnClass
 % nnInst is the instance of nnClass for this documentation
 
 % VARIABLES
 
     % 1) nnInst.nnState = 'train','validate' or 'test' : 'train' includes 'validate'
 
     % 2) nnInst.HIDDEN_ALPHA : ELU activation function. set alpha smaller, result is good.
     % 3) nnInst.HIDDEN_ALPHA : TANH activation function. alpha=2/x, if we want y==1 for x==1, alpha must be 2.
 
     % 4) nnInst.CHOQUET_NODE_SIZE = [hidden(1) hidden(2) ... hidden(n)]
     % 5) nnInst.NETWORK_NODE_SIZE = [hidden(1) hidden(2) ... hidden(n)]
 
     % 6) nnInst.ADAP_LEARN_PARAM : [MaxLearnRate MinLearnRate EpochPeriod] , in a
     % epoch period cosine annealing adaptive learning parameter will be
     % applied. for example [0.1 0.001 100] will apply 0.1 to 0.001 learning
     % rate with cosine annealing in each 100 epoch period, if total epoch
     % count is 1000, cosine annealing learning rate will repeat 10 times
     % from 0.1 to 0.001
 
     % 7) nnInst.MAX_EPOCH: maximum epoch count;
 
     % 8) nnInst.NORM_COEFF=[a b]; if you call nnInst.normalizeX()

     % 9) nnInst.epochOutputFolder : Output folder for periodic epoch counts, if it is not determined, there will be no epoch graphics
     % 10) nnInst.epochOutputPeriod : Period of saving outputs for periodic epoch counts
 
 % METHODS


 % x(m,n): input matrix with m input nodes n samples.
 % d(k,n): desired output matrix with k output nodes n samples.

 % y(k,n): output matrix with k output nodes n samples.

 % x_: all outputs of all nodes including input values.(Multilayer Perceptron Layers)
 % x_diff(i,n): all derivation outputs of all nodes including input values.(Multilayer Perceptron Layers)
 % z_: all outputs of all nodes including input values.(Choquet Layers)

     % 1) [y, x_diff, x_, z_] = nnInst.choqmlp(x)
 
     % 2) nnInst = nnInst.backpropagation(x, d)

    
    properties
        % adjustable parameters
        nnState='train'
        
        HIDDEN_ALPHA=0.1;
        OUTPUT_ALPHA=0.1;
        
        CHOQUET_NODE_SIZE=[];
        NETWORK_NODE_SIZE=[];
        MAX_EPOCH=1000
        
        ADAP_LEARN_PARAM=[0.01 0.05 0.3]

        NORM_COEFF=[1 0];
        
        epochOutputFolder=[]
        epochOutputPeriod=[]
        
        % information variables
        w=[]

        V=[]
        G=[]
        
        thr=0;
        
		e=[]
		trainPerf=[];
		

    end
    
    methods
        function x=normalizeX(nnObj, x)
            x=(x-nnObj.NORM_COEFF(:,1))./(eps+nnObj.NORM_COEFF(:,2)); %Standardization
        end
        
        function weights = init_weights(~,INPUT_SIZE,NODE_SIZE,rand_init)
            layerCount=size(NODE_SIZE,2);
            weights=cell(1,sum(NODE_SIZE));

            for i=1:layerCount
                if i==1
					stddev=sqrt(2/(INPUT_SIZE+1));
                    for j=1:NODE_SIZE(1)
                        weights{j}=rand_init*normrnd(0,stddev,1,(INPUT_SIZE+1));
                    end
                else
                    stddev=sqrt(2/(NODE_SIZE(i-1)+1));
                    for j=(sum(NODE_SIZE(1:(i-1)))+1):sum(NODE_SIZE(1:i))
                        weights{j}=rand_init*normrnd(0,stddev,1,(NODE_SIZE(i-1)+1));
                    end
                end
            end
        end
        
        function VGweights = init_VGweights(~,INPUT_SIZE,NODE_SIZE,rand_init)
            layerCount=size(NODE_SIZE,2);
            VGweights=cell(1,sum(NODE_SIZE));

            for i=1:layerCount
                if i==1
                    for j=1:NODE_SIZE(1)
                        VGweights{j}=rand_init*rand(1,INPUT_SIZE);
                    end
                else
                    for j=(sum(NODE_SIZE(1:(i-1)))+1):sum(NODE_SIZE(1:i))
                        VGweights{j}=rand_init*rand(1,NODE_SIZE(i-1));
                    end
                end
            end
        end
        
        function [y,x_diff,x_,z_diff,z_,inv_sigma] = choqmlp(nnObj,x_orig)
            %y: output array or value
            %x_diff: all derivation outputs of all nodes including input values.(Multilayer Perceptron Layers)
            %x_: all outputs of all nodes including input values.(Multilayer Perceptron Layers)
            %z_: all outputs of all nodes including input values.(Choquet Layers)
            %x_orig(i,n): ith input of nth sample
            
            %Choquet Integral Layers
            
            CHOQUET_INPUT_SIZE=size(x_orig,1);
            
            CHOQUET_NODE_SIZE_TEMP=nnObj.CHOQUET_NODE_SIZE;
            
            choquetLayerCount=size(nnObj.CHOQUET_NODE_SIZE,2);
            
            if choquetLayerCount>0
                [z,z_,z_diff,inv_sigma]=choquetnode_core_mex(choquetLayerCount,CHOQUET_NODE_SIZE_TEMP,CHOQUET_INPUT_SIZE,nnObj.HIDDEN_ALPHA,nnObj.V,nnObj.G,x_orig);
            else
                z_=[];
                z_diff=z_;
                z=x_orig;
                inv_sigma=[];
            end
            %Multilayer Perceptron Layers
            MLP_INPUT_SIZE=size(z,1);
            
            NETWORK_NODE_SIZE_TEMP=[nnObj.NETWORK_NODE_SIZE (size(nnObj.w,2)-sum(nnObj.NETWORK_NODE_SIZE))];
            
            layerCount=size(NETWORK_NODE_SIZE_TEMP,2); %get layer count

            [y,x_,x_diff]=multilayerperceptron_core_mex(layerCount,NETWORK_NODE_SIZE_TEMP,MLP_INPUT_SIZE,nnObj.HIDDEN_ALPHA,nnObj.OUTPUT_ALPHA,nnObj.w,z);
        end
        
        function nnObj=backpropagation(nnObj,x,d,val_x,val_d)
            rng('shuffle', 'simdTwister');
            
            %init Choquet W and G
            CHOQUET_INPUT_SIZE=size(x,1);
            CHOQUET_NODE_SIZE_TEMP=nnObj.CHOQUET_NODE_SIZE;

            nnObj.V = nnObj.init_VGweights(CHOQUET_INPUT_SIZE,CHOQUET_NODE_SIZE_TEMP,1);
            nnObj.G = nnObj.init_VGweights(CHOQUET_INPUT_SIZE,CHOQUET_NODE_SIZE_TEMP,1);
            
            choquetLayerCount=size(nnObj.CHOQUET_NODE_SIZE,2);

            %initalize x-add bias part- and w-random numbers-
            if choquetLayerCount>0
                MLP_INPUT_SIZE=CHOQUET_NODE_SIZE_TEMP(end);
            else
                MLP_INPUT_SIZE=size(x,1);
            end
            NETWORK_NODE_SIZE_TEMP=[nnObj.NETWORK_NODE_SIZE size(d,1)];
            
            nnObj.w = nnObj.init_weights(MLP_INPUT_SIZE,NETWORK_NODE_SIZE_TEMP,1);
            
            layerCount=size(NETWORK_NODE_SIZE_TEMP,2);
            
            %set learn rate and sample count
            M=size(x,2);
            nu_max=nnObj.ADAP_LEARN_PARAM(1);
            nu_min=nnObj.ADAP_LEARN_PARAM(2);
            nu_period=nnObj.ADAP_LEARN_PARAM(3);
            
            LEARN_RATE=nnObj.ADAP_LEARN_PARAM(1);
            
            %init epoch
			maxPerf=0;
            nnObjBestPerf=nnObj;
            val_y=nnObj.choqmlp(val_x);
            perf(1:nnObj.MAX_EPOCH)=nnObj.performance(val_d,val_y);
            truthTable=nnObj.classTruth(); 
            
            rng('shuffle', 'simdTwister');
            for k=1:nnObj.MAX_EPOCH %for an epoch
                [x, d, ~] = nnObj.shuffle( x, d, d(1,:));
                nnObj.e(k)=0; nnObj.thr=zeros(size(d,1),1);
                for n=1:M%for each sample
                    [y,x_diff,x_,z_diff,z_,inv_sigma]= nnObj.choqmlp(x(:,n));%x_(i,n): ith input of nth sample
                    %Check NaN
                    if max(isnan(y))==1 || max(isinf(y))==1
						nnObj.V = nnObj.init_VGweights(CHOQUET_INPUT_SIZE,CHOQUET_NODE_SIZE_TEMP,1);
						nnObj.G = nnObj.init_VGweights(CHOQUET_INPUT_SIZE,CHOQUET_NODE_SIZE_TEMP,1);
                        nnObj.w = nnObj.init_weights(MLP_INPUT_SIZE,NETWORK_NODE_SIZE_TEMP,1);
						LEARN_RATE=nnObj.ADAP_LEARN_PARAM(1);
                        [y,x_diff,x_,z_diff,z_,inv_sigma]= nnObj.choqmlp(x(:,n));%x_(i,n): ith input of nth sample
                    end
                    
                    err=-(y-d(:,n));
                    nnObj.e(k)=nnObj.e(k)+mean(abs(err)/(truthTable(2)-truthTable(1)))/M;
                    
                    %MLP Part
                    [nnObj.w, dirac]=backpropagation_core_mex(layerCount, NETWORK_NODE_SIZE_TEMP, MLP_INPUT_SIZE, err, x_diff, LEARN_RATE, x_, nnObj.w);
                    
                    %Choquet Part
                    if choquetLayerCount>0
                        [nnObj.V, nnObj.G]=choquetbackpropagation_core_mex(choquetLayerCount, NETWORK_NODE_SIZE_TEMP, CHOQUET_NODE_SIZE_TEMP, CHOQUET_INPUT_SIZE, z_diff, LEARN_RATE, z_, dirac, inv_sigma, nnObj.V, nnObj.G, nnObj.w);
                    end
                end
                
                y= nnObj.choqmlp(x);
                for i=1:size(d,1)
                    [X,Y,thr_all,~,optopr]=perfcurve(d(i,:),y(i,:),truthTable(2),'Prior','uniform','NegClass',truthTable(1));
                    nnObj.thr(i)=thr_all((X==optopr(1))&(Y==optopr(2)));
                end
                
				%Hold Best Result
				if k==1
					val_y=nnObj.choqmlp(val_x);
					perfVal=nnObj.performance(val_d,val_y);
                    maxPerf=mean(perfVal.accuracy);
					perf(k)=perfVal;
					nnObjBestPerf=nnObj;
				else
					val_y=nnObj.choqmlp(val_x);
					perfVal=nnObj.performance(val_d,val_y);
					if (perfVal.ROCperf>=maxPerf)
                        maxPerf=perfVal.ROCperf;
						perf(k)=perfVal;
						nnObjBestPerf=nnObj;
                    else
						perf(k)=perf(k-1);
%                         nnObjTemp=nnObjBestPerf;
%                         nnObjTemp.e=nnObj.e;
%                         nnObj=nnObjTemp;
					end
				end
                
                %Epoch Output
                if max(size(nnObj.epochOutputFolder))~=0
                    if k==1
                        if exist(nnObj.epochOutputFolder,'dir')==0
                            mkdir(nnObj.epochOutputFolder);
                        else
                            rmpath(genpath(nnObj.epochOutputFolder));
                            rmdir(nnObj.epochOutputFolder,'s');
                            mkdir(nnObj.epochOutputFolder);
                        end
                    end
                    if mod(k,nnObj.epochOutputPeriod)==0
                        nnObj.epochPerf(k,perf,nnObjBestPerf); clc;
                    end
                end
				
                %Learning Rate Update
                LEARN_RATE=nu_min + (1/2)*(nu_max-nu_min)*(1+cos((k/nu_period)*pi));
                if mod(k,nu_period)==0
                    nnObj.V = nnObj.init_VGweights(CHOQUET_INPUT_SIZE,CHOQUET_NODE_SIZE_TEMP,1);
                    nnObj.G = nnObj.init_VGweights(CHOQUET_INPUT_SIZE,CHOQUET_NODE_SIZE_TEMP,1);
                    nnObj.w = nnObj.init_weights(MLP_INPUT_SIZE,NETWORK_NODE_SIZE_TEMP,1);
                end
            end
            nnObj=nnObjBestPerf;
        end
        
        function epochPerf(nnObj,k,perf,nnObjBest)
            k_outputFolder=[nnObj.epochOutputFolder '\nnfiles'];
			if k==nnObj.epochOutputPeriod
				if exist(k_outputFolder,'dir')==0
					mkdir(k_outputFolder);
				else
					rmdir(k_outputFolder,'s');
					mkdir(k_outputFolder);
				end
			end
			
			errorGraphFile=[nnObj.epochOutputFolder '/Training_errorGraph'];
            perfGraphFile=[nnObj.epochOutputFolder '/Validation_perfGraph'];
            nnFile=[k_outputFolder '\epoch' num2str(k,['%0' num2str(1+floor(log10(nnObj.MAX_EPOCH))) 'd']) '_net_mlp.mat'];
            
            perf=struct2cell(perf);
            truPos=perf(1,:)';
            falPos=perf(2,:)';
            accUracy=perf(3,:)';
            truPos=cell2mat(truPos);
            falPos=cell2mat(falPos);
            accUracy=cell2mat(accUracy);
            
            f=figure('Name','Measured Data','NumberTitle','off','units','normalized','outerposition',[0 0 1 1],'DefaultAxesPosition', [0.1, 0.1, 0.8, 0.8]);
            set(f, 'Toolbar', 'none', 'Menu', 'none');
            set(f, 'Visible', 'off');
            set(f, 'Renderer', 'opengl');
            set(f, 'PaperUnits','normalized');

			yyaxis right;
            plot(1:k,nnObj.e(1:k)*100);
			yyaxis left;
            plot(1:k,nnObj.e(1:k)*100);
			grid minor; axis tight;
            title(['Mean Squared Error-Epoch Graph (Epoch ' num2str(k) ')'])
            xlabel('Epoch');
            ylabel('Mean Squared Error (%)');
            print(f,errorGraphFile,'-dpng');
            
            clf(f)
            for i=1:size(accUracy,2)
                subplot(size(accUracy,2),1,i)
				yyaxis right;
                plot(1:k,accUracy(1:k,i)*100);
				ylim([0 100]);
				yyaxis left;
                plot(1:k,accUracy(1:k,i)*100);
				grid minor; axis tight; ylim([0 100]);
                title(['Output ' num2str(i) ' (Epoch: ' num2str(k) ', Accuracy: ' num2str(accUracy(k,i)*100) '%) '])
                xlabel('Epoch');
                if i==1
                    ylabel('Validation Classification Performance (%)');
                end
            end
            print(f,perfGraphFile,'-dpng');
			
            nnObject=nnObjBest;
            save(nnFile,'nnObject');
        end
        
        function labels=labeledOutput(nnObj,classTypes,y)
            y_tresh_bin=y>=nnObj.thr;
            y_tresh_dec=zeros(1,size(y_tresh_bin,2));
            for i=1:size(y_tresh_bin,1)
                y_tresh_dec=y_tresh_dec+y_tresh_bin(i,:).*i;
            end
            labels=cell(1,size(y_tresh_bin,2));
            for i=1:size(y_tresh_dec,2)
                if (y_tresh_dec(i)<=size(classTypes,2)) && (y_tresh_dec(i)>0)
                    labels{i}=classTypes{y_tresh_dec(i)};
                else
                    labels{i}=' '; 
                end
            end
        end
        
        function perf = performance(nnObj,d,y)
            class_truth=nnObj.classTruth();
            
            AUC=zeros(size(d,1),1);
            for i=1:size(d,1)
                [~,~,~,AUC(i),~]=perfcurve(d(i,:),y(i,:),class_truth(2),'Prior','uniform','NegClass',class_truth(1));
            end
            
            y_thres=y;
            y_thres(y_thres<=nnObj.thr)=0;
            [~,y_ind]=max(y_thres,[],1);
            y_thres=y_thres.*0;
            for i=1:size(y_ind,2)
                y_thres(y_ind(i),i)=1;
            end
            
            y_thres=(class_truth(2)-class_truth(1)).*y_thres+class_truth(1);
            y_comp=(class_truth(2)-class_truth(1)).*eye(size(d,1))+class_truth(1);
            
            true_positive=zeros(1,size(d,1));   true_negative=zeros(1,size(d,1));
            false_positive=zeros(1,size(d,1));  false_negative=zeros(1,size(d,1));

            for i=1:size(d,1)
                y_cmp=min(y_thres==y_comp(:,i));
                d_cmp=min(d==y_comp(:,i));
                true_positive(i)=sum((y_cmp) & (d_cmp));
                false_positive(i)=sum((y_cmp) & (~d_cmp));
                false_negative(i)=sum((~y_cmp) & (d_cmp));
            end
            for i=1:size(d,1)
                true_negative(i)=sum(true_positive)-true_positive(i);
            end
            
            perf.truePositiveRate=true_positive./(true_positive+false_negative);
            perf.falsePositiveRate=false_positive./(false_positive+true_negative);
            perf.accuracy=(true_positive+true_negative)./(true_positive+false_negative+false_positive+true_negative);
            
            perf.AUC_mu=mean(AUC);
            perf.AUC_std=std(AUC);
            
            perf.ROCperf=mean((perf.truePositiveRate+(1-perf.falsePositiveRate))/2);
        end
        
        function [output1, output2, output3] = shuffle(~, input1, input2, input3 )
            output1=input1;
            output2=input2;
            output3=input3;
            
            rprm=randperm(size(input1,2));
            
            i=1:size(input1,1);
            j=1:size(input2,1);
            
            output1(i,rprm) = input1(i,:);
            output2(j,rprm) = input2(j,:);
            output3(rprm) = input3(:);
        end
        
        function [output] = shuffleArray(~, input)
            output=input;
            
            rprm=randperm(size(input,2));
            
            i=1:size(input,1);
            
            output(i,rprm) = input(i,:);
        end
        
        function truthTable=classTruth(~)
            truthTable=[-1 1];
        end
    end
end

