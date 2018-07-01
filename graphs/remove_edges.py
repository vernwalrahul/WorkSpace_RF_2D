import networkx as nx
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate environments')
    parser.add_argument('--graphfile',type=str,required=True)
    args = parser.parse_args()

    G = nx.read_graphml(args.graphfile)
    G.remove_edges_from(list(G.edges()))

    nx.write_graphml(G, args.graphfile)

if __name__ == '__main__':
    main()