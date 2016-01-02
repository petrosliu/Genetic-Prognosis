function data=readdata(filename)
    tempdata=importdata(filename);
    [~,n]=size(tempdata.data);
    data.patientID=tempdata.data(:,1);
    data.gene=tempdata.data(:,2:n-1);
    data.label=tempdata.data(:,n);
    data.genename=tempdata.textdata(2:71);
end