from data_generator_v2 import DataGenerator

obj= DataGenerator()
data,data_org=obj.train_batch(
    batch_size=4,
    prob_p=0.4,
    prob_q=0.5,
    num_negative_samples=3,
    max_length=12,
    num_walks=10,
    dimension=12,
    walk_max_length=5,
    x_dim=4,
    y_dim=3,
    embedding_dim=128,
    num_epochs=100
)
print(len(data))
print(len(data[0]))
print(len(data[0][0]))
# [bs,num_nodes,embedding_dim]