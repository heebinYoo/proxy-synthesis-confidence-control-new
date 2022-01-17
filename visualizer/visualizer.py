import numpy as np
import torch
import torch.nn.functional as F

import faiss
import seaborn as sns
import colorcet as cc
# see : https://docs.cupy.dev/en/stable/install.html
import cupy as cp
from matplotlib import pyplot as plt
from cuml.manifold import TSNE
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def get_embed(model, pooling, embedding, device, dataloader, feature_dim, batch_size, number_of_data, simple=False):
    embedding_list = np.zeros(shape=(number_of_data, feature_dim))
    label_list = np.zeros(shape=number_of_data) - 2

    iter_num = 0
    for inputs, labels in tqdm(dataloader, dynamic_ncols=True):
        with torch.no_grad():
            outputs = model(inputs.to(device))
            if not simple:
                outputs = pooling(outputs)
                outputs = embedding(outputs)

        labels = labels.detach().cpu().numpy().ravel()
        number_in_batch = labels.shape[0]
        label_list[iter_num * batch_size:iter_num * batch_size + number_in_batch] = labels
        embedding_list[iter_num * batch_size:iter_num * batch_size + number_in_batch] = outputs.detach().cpu().numpy()

        iter_num += 1
    if not np.where(label_list == -2)[0].shape[0] == 0:
        embedding_list = embedding_list[:np.min(np.where(label_list == -2)), :]
        label_list = label_list[:np.min(np.where(label_list == -2))]

    return embedding_list, label_list

    # if set_two_color:
    #     y[np.where(np.logical_and(y >= 0, y < number_of_train_class))] = 0
    #     y[np.where(np.logical_and(y >= number_of_train_class, y <= number_of_train_class + number_of_test_class))] = 1
    #
    # if exclude_weight:
    #     data = data[:-number_of_train_class]
    #     y = y[:-number_of_train_class]
    #     size = np.ones_like(y) + 2
    # else:


def generate_weight_tsne_figure(train_embedding_list, train_label_list, test_embedding_list, test_label_list,
                                number_of_train_class, number_of_test_class, final_weight):
    # integrated_test_label_list = test_label_list
    data = test_embedding_list  # np.concatenate((test_embedding_list, final_weight), axis=0)
    weight_label = np.array(
        [a + number_of_test_class for a in range(number_of_train_class)])
    data = TSNE(n_components=2, perplexity=30, n_neighbors=500, learning_rate=150).fit_transform(data)

    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.set_style('darkgrid')

    y = test_label_list  # np.concatenate((integrated_test_label_list, weight_label))

    size = np.ones_like(y) + 2
    #size[-number_of_train_class:] = 45

    palette = sns.color_palette(cc.glasbey, n_colors=np.unique(y).shape[0])

    sns.scatterplot(x=data[:, 0], y=data[:, 1], s=size, hue=y, legend=False, palette=palette)
    return fig


def plot_mnist_embedding(num_class, train_embedding_list, train_label_list, test_embedding_list, test_label_list):
    train_color = np.zeros_like(train_label_list, dtype='str')
    for i, c in zip(range(num_class), ['r', 'g', 'b', 'y', 'm', 'c']):
        train_color[np.where(train_label_list == i)] = c

    embed_fig = plt.figure(figsize=(10, 10))
    plt.scatter(train_embedding_list[:, 0], train_embedding_list[:, 1], alpha=0.6, c=train_color, s=30)
    plt.scatter(test_embedding_list[np.where(test_label_list == 5), 0],
                test_embedding_list[np.where(test_label_list == 5), 1], alpha=0.8, c='#222222', s=10)

    plt.scatter(test_embedding_list[np.where(test_label_list == 6), 0],
                test_embedding_list[np.where(test_label_list == 6), 1], alpha=0.8, c='#343434', s=10)

    plt.title("embedding space")
    # plt.show()
    return embed_fig


def plot_synthesis_input(model, device, feature_dim, train_data_loader, test_data_loader):
    # input

    train_color = np.zeros_like(train_data_loader.dataset.y_data, dtype='str')
    train_color[np.where(train_data_loader.dataset.y_data == 0)] = 'b'
    train_color[np.where(train_data_loader.dataset.y_data == 1)] = 'r'

    input_fig = plt.figure(figsize=(10, 10))
    plt.scatter(train_data_loader.dataset.x_data[:, 0], train_data_loader.dataset.x_data[:, 1], c=train_color, s=10)
    plt.scatter(test_data_loader.dataset.x_data[:, 0], test_data_loader.dataset.x_data[:, 1], alpha=0.3, c='#222222',
                s=10)

    xi = np.linspace(np.min(train_data_loader.dataset.x_data[:, 0]), np.max(train_data_loader.dataset.x_data[:, 0]),
                     100)
    yi = np.linspace(np.min(train_data_loader.dataset.x_data[:, 1]), np.max(train_data_loader.dataset.x_data[:, 1]),
                     100)

    X, Y = np.meshgrid(xi, yi)

    zi = model.forward(
        torch.FloatTensor(
            np.concatenate(
                (X.reshape((-1, 1)), Y.reshape((-1, 1))),
                axis=1)
        ).reshape(-1, feature_dim).to(device)
    ).detach().cpu().numpy()

    zi_softmax = F.softmax(torch.from_numpy(zi), dim=1).numpy()

    zi_argmax_sign = np.argmax(zi, axis=1)
    zi_argmax_sign[np.where(zi_argmax_sign == 0)] = -1

    Z = (np.max(zi_softmax, axis=1) * zi_argmax_sign).reshape(X.shape)

    plt.contourf(X, Y, Z, levels=15, alpha=0.3, cmap="bwr")
    plt.colorbar()
    plt.title("input space")
    return input_fig
    # plt.show()


def plot_synthesis_embedding(model, device, feature_dim, train_embedding_list, train_label_list, test_embedding_list,
                             test_label_list):
    train_color = np.zeros_like(train_label_list, dtype='str')
    train_color[np.where(train_label_list == 0)] = 'b'
    train_color[np.where(train_label_list == 1)] = 'r'

    embed_fig = plt.figure(figsize=(10, 10))
    plt.scatter(train_embedding_list[:, 0], train_embedding_list[:, 1], c=train_color, s=10)
    plt.scatter(test_embedding_list[:, 0], test_embedding_list[:, 1], alpha=0.6, c='#222222', s=10)

    xi = np.linspace(np.min(train_embedding_list[:, 0]), np.max(train_embedding_list[:, 0]), 100)
    yi = np.linspace(np.min(train_embedding_list[:, 1]), np.max(train_embedding_list[:, 1]), 100)

    X, Y = np.meshgrid(xi, yi)

    zi = model.forward(
        torch.FloatTensor(
            np.concatenate(
                (X.reshape((-1, 1)), Y.reshape((-1, 1))),
                axis=1)
        ).reshape(-1, feature_dim).to(device)
    ).detach().cpu().numpy()
    zi_softmax = F.softmax(torch.from_numpy(zi), dim=1).numpy()

    zi_argmax_sign = np.argmax(zi, axis=1)
    zi_argmax_sign[np.where(zi_argmax_sign == 0)] = -1

    Z = (np.max(zi_softmax, axis=1) * zi_argmax_sign).reshape(X.shape)

    plt.contourf(X, Y, Z, levels=15, alpha=0.3, cmap="bwr")
    plt.colorbar()
    plt.title("embedding space")
    # plt.show()

    return embed_fig


def generate_weight_embedding_relation_heatmap_figure(embedding_data_list, final_weights, feature_dim,
                                                      embedding_label_list):
    label_list = np.unique(embedding_label_list)
    final_weights = final_weights[label_list.astype('int')]

    embedding_mean_per_class = np.zeros(shape=(label_list.shape[0], feature_dim))
    for i, c in enumerate(label_list):
        emb = embedding_data_list[np.where(embedding_label_list == c)]
        embedding_mean_per_class[i] = np.mean(emb, axis=0).reshape(-1)

    # embedding_mean_per_class = cp.asarray(embedding_mean_per_class)
    # embedding_mean_per_class_norm = cp.linalg.norm(embedding_mean_per_class, ord=2, axis=1)
    # normed_embedding_mean_per_class = embedding_mean_per_class / cp.expand_dims(embedding_mean_per_class_norm, axis=-1)
    #
    # final_weights = cp.asarray(final_weights)
    # final_weights_norm = cp.linalg.norm(final_weights, ord=2, axis=1)
    # normed_final_weights = final_weights / cp.expand_dims(final_weights_norm, axis=-1)
    #
    # index = faiss.IndexFlatIP(feature_dim)
    # index.add(normed_embedding_mean_per_class.astype('float32').get())
    # dists, _ = index.search(normed_final_weights.astype('float32').get(), label_list.shape[0])
    # dists = 1 - dists

    C = pairwise_distances(final_weights, embedding_mean_per_class, metric="cosine", n_jobs=4)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(C, cmap='jet', vmin=np.min(C), vmax=np.max(C))
    plt.colorbar()
    # plt.show()

    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(dists, cmap='jet', vmin=np.min(dists), vmax=np.max(dists))
    # plt.colorbar()
    return fig


def generate_feature_radius_dist_fig(train_embedding_list, test_embedding_list):
    fig = plt.figure(figsize=(10, 10))
    rad_test = np.linalg.norm(test_embedding_list, ord=2, axis=1)
    rad_train = np.linalg.norm(train_embedding_list, ord=2, axis=1)
    plt.xlim(0, 100)
    plt.ylim(0.0, 1)
    sns.kdeplot(rad_train, label="train", shade=True)
    sns.kdeplot(rad_test, label="test", shade=True)
    plt.xlabel("radius i.e. confidence")
    plt.legend()
    # plt.show()

    return fig


def generate_singular_value_figure(train_embedding_list, train_label_list, test_embedding_list, test_label_list):
    train_embedding_list = cp.asarray(train_embedding_list)
    train_embedding_norm = cp.linalg.norm(train_embedding_list, ord=2, axis=1)
    normed_train_embedding_list = train_embedding_list / cp.expand_dims(train_embedding_norm, axis=-1)

    test_embedding_list = cp.asarray(test_embedding_list)
    test_embedding_norm = cp.linalg.norm(test_embedding_list, ord=2, axis=1)
    normed_test_embedding_list = test_embedding_list / cp.expand_dims(test_embedding_norm, axis=-1)

    fig = plt.figure(figsize=(10, 10))
    for c in np.unique(train_label_list):
        idx = np.where(train_label_list == c)
        emb = normed_train_embedding_list[idx]
        s = cp.linalg.svd(emb, full_matrices=False, compute_uv=False)
        plt.plot(s.get(), c='b')

    for c in np.unique(test_label_list):
        idx = np.where(test_label_list == c)
        emb = normed_test_embedding_list[idx]
        s = cp.linalg.svd(emb, full_matrices=False, compute_uv=False)
        plt.plot(s.get(), c='r')

    plt.plot([], c='b', label='train')
    plt.plot([], c='r', label='test')
    plt.legend()
    return fig
