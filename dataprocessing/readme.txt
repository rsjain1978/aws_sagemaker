To prepare for training, you can preprocess your data using a variety of AWS services, including 
- AWS Glue, 
- Amazon EMR, 
- Amazon Redshift, 
- Amazon Relational Database Service, and 
- Amazon Athena. 

After preprocessing, publish the data to an Amazon S3 bucket. 

While the data is pushed to S3, the data coule be stored in either csv or recordio protobuf format. 

Most Amazon SageMaker algorithms work best when you use the optimized protobuf recordIO format for the training data. Using this format allows you to take advantage of Pipe mode when training the algorithms that support it. File mode loads all of your data from Amazon Simple Storage Service (Amazon S3) to the training instance volumes. In Pipe mode, your training job streams data directly from Amazon S3. Streaming can provide faster start times for training jobs and better throughput. With Pipe mode, you also reduce the size of the Amazon Elastic Block Store volumes for your training instances. Pipe mode needs only enough disk space to store your final model artifacts.

AWS supports Pipe mode for both CSV and RecordIO format files

Once the data is pushed into S3, SageMaker could be leverage to train and deploy the model. Model could be developed using SageMaker's Algorithms or using Algorithms as implemented in other frameworks/libraries.

Good Documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html