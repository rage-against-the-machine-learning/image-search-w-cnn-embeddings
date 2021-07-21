# Test Data
curl http://images.cocodataset.org/zips/test2014.zip > test2014.zip && \
unzip test2014.zip     && \
rm test2014.zip        && \

# Test Labels
curl http://images.cocodataset.org/annotations/image_info_test2014.zip > image_info_test2014.zip  && \
unzip image_info_test2014.zip    && \
rm image_info_test2014.zip