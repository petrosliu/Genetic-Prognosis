function [cnt,out]=savedata(selectedGene,index,erre,e,t,...
                                Lrn,Est,Srch,Sn)
    length=size(selectedGene(index,:),2);
    out(1,1:4)=[Lrn,Est,Srch,Sn];
    out(1,5:(5+length-1))=selectedGene(index,:);
    out(1,(5+length):9)=0;
    out(1,10:12)=[erre,e,t];
    disp(out);
    cnt=(Est-1)*24+(Lrn-1)*8+(Srch-1)*3+Sn;
    %mail2me('trigger@recipe.ifttt.com',num2str([Lrn,Est,Srch,Sn]),num2str([erre,e,t]));
end
