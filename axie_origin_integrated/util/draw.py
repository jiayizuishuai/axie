import plotext as plt


def terminal_draw(data_list, title):
    plt.clt()
    plt.cld()
    plt.plot(data_list)
    plt.title(title)
    plt.xlim(0, len(data_list)+1)
    low = min(data_list) if max(data_list) != min(
        data_list) else min(data_list) - 1
    plt.ylim(low, max(data_list))
    plt.plot_size(60, 20)
    plt.show()
