function savefile(outstream,genename,filename)
    data(1,:)={'classification rule','error estimator',...
                'feature selection','No.',1,[],2,[],3,[],4,[],5,[],...
                'error estimate','hold-out estimate','time'};
    for i=1:size(outstream,1)
        data(i+1,:)={[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]};
        switch outstream(i,1)
            case 1
                data(i+1,1)={'DiaLDA'};
            case 2
                data(i+1,1)={'3NN'};
            case 3
                data(i+1,1)={'LinSVM'};
        end
        
        switch outstream(i,2)
            case 1
                data(i+1,2)={'Resub'};
            case 2
                data(i+1,2)={'LOO'};
        end
        
        switch outstream(i,3)
            case 1
                data(i+1,3)={'Exhaustive'};
            case 2
                data(i+1,3)={'SFS'};
        end
         data(i+1,4)={uint8(outstream(i,4))};
        
        gene=uint8(zeros(1,5));
        gene(1:outstream(i,4))=uint8(sort(...
                                outstream(i,5:4+outstream(i,4))...
                                ));
        for j=1:5
            
            if gene(j)==0
                data(i+1,3+2*j)={[]};
                data(i+1,4+2*j)={[]};
            else
                data(i+1,3+2*j)={gene(j)};
                data(i+1,4+2*j)={genename(gene(j))};
            end
        end
        
        data(i+1,15)={outstream(i,10)};
        data(i+1,16)={outstream(i,11)};
        data(i+1,17)={outstream(i,12)};
    end
    save(filename,'data');
    disp(data);
end