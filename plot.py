def plot(G):
    nk = {}
    for _, k in G.degree():
        if k not in nk:
            nk[k] = 0
        nk[k] += 1
    ks = sorted(nk.keys())

    plt.loglog(ks, [nk[k] / len(G) for k in ks], '*k')
    plt.title(G.name)
    plt.ylabel('$p_k$')
    plt.xlabel('$k$')
    plt.show()


def plot(Gs, Ds, label):
    fig = plt.figure()

    plt.imshow(Ds, cmap=LinearSegmentedColormap.from_list('', ['yellow', 'gray', 'white']))
    for i in range(len(Gs)):
        for j in range(len(Gs)):
            plt.text(j, i, "{:.2f}".format(Ds[i][j]), ha='center', va='center', fontsize=4)

    plt.title(label)
    plt.xticks(ticks=[])
    plt.yticks(ticks=range(len(Gs)), labels=[G.name for G in Gs], fontsize=7)

    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=7)

    fig.savefig(label + ".pdf", bbox_inches='tight')
    plt.close(fig)