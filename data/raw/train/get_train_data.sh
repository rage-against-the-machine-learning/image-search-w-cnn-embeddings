# Training Data
curl http://images.cocodataset.org/zips/train2014.zip > train2014.zip && \
unzip train2014.zip     && \
rm train2014.zip        && \

# Training Labels
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip > train_val_labels.zip  && \
unzip train_val_labels.zip    && \
rm train_val_labels.zip
