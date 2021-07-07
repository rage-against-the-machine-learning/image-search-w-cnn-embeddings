# Validation Data
mkdir /validation  && \
cd /validation     && \
curl http://images.cocodataset.org/zips/val2014.zip > val2014.zip && \
unzip val2014.zip     && \
rm val2014.zip