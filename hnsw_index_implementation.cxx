#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iomanip>   // std::fixed / std::setprecision
using namespace std;

/* ========================================================
 * Printing out Data Points
 * Loading fvecs and ivecs
 * ========================================================
*/
ostream& operator<<(ostream& os, const vector<float>& v) { // << operator definition
														   // Helper for pretty-printing a single vector
														   // - so we can print out and visualize a data point
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        os << fixed << setprecision(5) << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << ']';
    return os;
} // end of << operator definition

vector<vector<float>> load_fvecs(const string& filename) { // load_fvecs definition
														   // used to load the data files into a variable
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open file " + filename);

    vector<vector<float>> data;

    // Read loop: first try to read the dim; if that succeeds, read the payload.
    for (;;) {
        int dim = 0;
        if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int)))
            break;                       // normal EOF

        if (dim <= 0 || dim > 1'000'000) // crude sanity check
            throw runtime_error("Suspicious vector dimension: " + to_string(dim));

        vector<float> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float)))
            throw runtime_error("Truncated file when reading vector data.");

        data.push_back(move(vec));
    }

    return data; 
} // end of load_fvecs definition

vector<vector<int>> load_ivecs(const string& filename){ // load i_vecs definition
													    // used to load data file into a variable
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file " + filename);

    std::vector<std::vector<int>> data;

    for (;;) {
        int dim = 0;
        if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int)))
            break;                              // clean EOF

        if (dim <= 0 || dim > 1'000'000)
            throw std::runtime_error("Suspicious vector dimension: " +
                                     std::to_string(dim));

        std::vector<int> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()),
                       dim * sizeof(int)))
            throw std::runtime_error("Truncated file while reading vector.");

        data.push_back(std::move(vec));
    }
    return data;
} // end of load_ivecs definition

/* ===========================================================
 * HNSW Index Implementation
 * ===========================================================
*/

typedef vector<float> Point; // a point in d-dimensional space
							  // giving an alias to the type vector<double> by calling it "Point
struct Node{ // node struct definition
			 // a node is an element in the HNSW graph
	
	// attributes of Node
	Point data;
	int level;
	vector<vector<int>> neighbors;
	
	Node(const Point &d, int l) : data(d), level(l), neighbors(l+1){} // Node constructor

}; // end of Node struct definition

class HNSW{  // HNSW Index class definition
	
	public:
		
		HNSW(int M, int Mmax, int Mmax0, int efC, double mL) : M(M), Mmax(Mmax), Mmax0(Mmax0), efConstruction(efC), mL(mL), enter_point(-1),max_level(-1){} // HNSW's Constructor
		
		void insert(const Point &q){ // insert(.) method definition
									 // Algorithm1
			
			// generate random level l for the new element (element to be inserted)
			double r = uniform_dist(rng);
			int l = (int)floor(-log(r)*mL);
			
			
			int id = (int)nodes.size(); // ID of the new node is its index in 'nodes' <--- a vector holding all of the nodes. It's an attribute of HNSW
			nodes.emplace_back(q,l); // construct a new node and put it into the vector called 'nodes'
			
			
			// Only do this for the first insertion ---- initialize entry point
			if(enter_point < 0){ // enter_point is initialized to -1 so if it's still -1 then that means we're at our first insertion
				enter_point = id; // set first inserted node as entry point of the entire HNSW graph
				max_level = l; // set l to be highest layer of the HNSW 
			}
			
			
			int ep = enter_point; // current entry point
			int L = max_level; // highest layer in the graph
			
			// Phase 1: search down from the top layer to layer (l+1) with ef=1
			// express search
			for(int lc = L; lc > l; --lc){
				vector<int> W = searchLayer(q, ep, 1, lc); // W is a list containing the nearest nodes to q that live on this current layer
				ep = getNearest(q,W); // the entry point for the upcoming layer (one beneath this layer) will be the closest node to q found in this layer	                      
			}
			
			
			// Phase 2: search from layer min(L,l) down to 0 with ef=efConstruction
			// more detailed search
			for(int lc = min(L,l); lc >= 0; --lc){
				
				vector<int> W = searchLayer(q, ep, efConstruction,lc);
				vector<int> neighbors = selectNeighborsSimple(q,W,M); // user gets to pick which algorithm to use to choose neighbors for q
																	  // Algorithm3 is selectNeighborsSimple
																	  // Algorithm4 is selectNeighborsHeuristic
				
				
				for(int n : neighbors){ // add bidirectional links between neighbor n and you q
					
					nodes[id].neighbors[lc].push_back(n);
					nodes[n].neighbors[lc].push_back(id);
				}
				
				
				
				for(int n : neighbors){ // scan through each neighbor of q, call 'each neighbor' n, and shrink each neighbor's outward links if they exceed capacity
					
					auto &nConn = nodes[n].neighbors[lc]; // nConn is a vector containing n's neighbors at level lc
					int cap = (lc == 0 ? Mmax0 : Mmax); // cap (aka limit) depends on which level we're on
															// if we're on ground level then we have a different cap than if we're on non-ground level
					if((int)nConn.size() > cap){ // if n has too many neighbors ....
						vector<int> nNewConn = selectNeighborsSimple(nodes[n].data,nConn,cap); // ... we must filter some neighbors and only retain the 'best' neighbors for n
						nConn.swap(nNewConn);
					}
				}
				
				ep = getNearest(q,W); // out of all the good candidates you just found at layer lc, pick the best node that's closest to q and use that 'best node' as your entry point for when you drop down to the next layer
			}
				
				
			if(l > max_level){ // if newly inserted node is present at the highest level then update entry point to the entire HNSW graph object 
				enter_point = id;
				max_level = l;
			}
				
			
		} // end of insert(.) method definition
		
		vector<int> searchKNN(const Point &q, int k, int ef){ // searchKNN(.) method definition
															  // Algorithm5
															  
					vector<int> W;
					int ep = enter_point; // ep takes on the entry point of the entire HNSW graph
					int L = nodes[ep].level; // L takes on the top layer of the HNSW graph
					
					// KEY: searching in HNSW is insertion with q's layer being 0
					
					// Greedy search from top layer down to layer1 with ef=1
					// Phase 1
					for(int lc = L; lc >= 1; --lc){
						W = searchLayer(q,ep,1,lc);
						ep = getNearest(q,W);
					}
					
					// Phase2
					// final search on the ground layer
					W = searchLayer(q,ep,ef,0);
					
				
					// out of all the nodes in W, return topK closest nodes to q
					sort(W.begin(), W.end(), [&](int a, int b){ return distance(q,nodes[a].data) < distance(q, nodes[b].data);}); // sort all nodes in W based on distance of each node to q
					if((int)W.size() > k) { W.resize(k);}
					return W;
		
	    } // end of searchKNN method definition
	    
	    
	    const Point& pointGetter(int pointID) const { // pointGetter definition
													  // so main() can actually view the data point
									
			if (pointID < 0 || pointID >= static_cast<int>(nodes.size())){ throw out_of_range("invalid node id");} // error guard
			
			return nodes[pointID].data; // return a Point object
				
		} // end of pointGetter definition
		
	private:
		
		// attributes of HNSW
		int M; // number of connections per node (aka per insertion)
		int Mmax; // maximum number of connections per non-ground layer node
		int Mmax0; // maximum number of connections per ground layer node
		int efConstruction; // ef parameter during insertion
		double mL; // level normalization
		int enter_point; // current entry point
		int max_level; // current maximum layer index
		vector<Node> nodes; // all nodes in the HNSW structure
		// RNG for level selection
		mt19937 rng{random_device{}()};
		uniform_real_distribution<double> uniform_dist{numeric_limits<double>::min(),1.0};
		
		double distance(const Point &a, const Point &b){ // distance(.) method definition
													     // Euclidean distance
			double sum = 0;
			for(size_t i=0; i<a.size(); ++i){
				sum += (a[i] - b[i]) * (a[i] - b[i]);
			}
			return sqrt(sum);
		} // end of distance(.) method definition
		
		vector<int> searchLayer(const Point &q, int ep, int ef, int lc){ // searchLayer(.) method definition
			
			// preliminaries
			unordered_set<int> visited; // track nodes we've already looked at to avoid revisits
			visited.insert(ep); // immediately mark entry point as seen 
			
			priority_queue<pair<double,int>, vector<pair<double,int>>,greater<>> C; // C is a candidate set (ie a min-heap ordered by distance)
																					// pair is (distance to q, nodeID)
																					// vector is the container
																					// greater<> is the min-heap comparator
																					// NOTE: we'll always pull the closest unseen candidate next
																					
			priority_queue<pair<double,int>> W; // is window (aka beam) of current top-ef best results (max-heap ordered by distance)
			                                    // maintain up to ef amount of closest nodes seen so far; top() is the worst of those
			                                    // pair is (distance to q, nodeID)
			                                    // this is our dynamic list of found nearest neighbors so far
			
			double d0 = distance(q, nodes[ep].data); // compute distance from entry point to q
			C.emplace(d0,ep); // initializations to get things started
			W.emplace(d0,ep);
			
			
			
			while(!C.empty()){ // keep expanding candidates until no promising candidates remain
				
				// take the closest unexpanded candidate
				auto [d_c, c] = C.top(); // d_c is distance from node c to q
				                         // c is the nodeID
				C.pop();
				
				// take the worst (furthest) element in our current best window (beam)
				auto [d_f, f] = W.top(); // f is the nodeID
				                         // d_f is the distance between node f and node q
				                         
				if(d_c > d_f){break;} // if the closest candidate to q is already further than our worst current best there's no way we'll find a better neighbor by expanding c
				
				// otherwise, expand node c's neighbors at layer lc
				for(int e : nodes[c].neighbors[lc]){
					
					// e is the ID of one neighbor of c on this layer
					
					if(visited.count(e)){continue;} // skip this node if we've already visited
													// move onto the next neighbor
													
					visited.insert(e); // mark e as visited
					
					double d_e = distance(q, nodes[e].data); // compute the distance from query to this neighbor e
					
					
					
					// if our 'current-best' window W isn't full yet or e is closer than the worst in W, then e is worth exploring
					
					if((int) W.size() < ef || d_e < W.top().first){
						
						C.emplace(d_e, e); // put e into the candidate set so we can expand it in the future
						
						W.emplace(d_e,e); // put e into our 'current best' window W;
						
						if((int) W.size() > ef){W.pop();} // if we exceed window capacity, ef, then remove the worst element from W
					}
				}
			}
			
			
			// extract final results from W into a simple vector of IDs
			vector<int> results;
			while(!W.empty()){
				results.push_back(W.top().second);
				W.pop();
			}
			return results;
		
		} // end of searchLayer(.) method definition
		
		vector<int> selectNeighborsSimple(const Point &q, const vector<int> &C, int M){ // selectNeighborsSimple method definition
																						// Algorithm3																			
			// Parameters:
			// -     q is the target node
			// - 	 C is a list of candidates nodes that can be q's neighbors
			// -     M is how many neighbors q is allowed to get																		
			
			vector<int> Csorted = C;
			
			sort(Csorted.begin(), Csorted.end(), [&](int a, int b){ return distance(q, nodes[a].data) < distance(q, nodes[b].data);});
			if((int)Csorted.size() > M){Csorted.resize(M);}
			return Csorted;
		} // end of selectNeighborsSimple method definition
		
		vector<int> selectNeighborsHeuristic(const Point &q, const vector<int> &C, int M, int lc, bool extendCandidates, bool keepPrunedConnections) { // selectNeighborsHeuristic method definition
																																					   // Algorithm4
			
			/* the preliminaries */
			vector<int> R; // R contains final selected neighbors
						   // R for results
			priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> W, Wd; // W is a min heap keyed on distance(q,e)
																						 //  	W is 'working set'
																						 // Wd is a queue for pruned / discarded candidates
																						 //  	Wd is 'working discarded'
			unordered_set<int> inW; // helper set to avoid duplicates in W
									// inW tracks which nodes are already inside of W
			
			
			
			for(int c : C){ // initialize W with every node c in C
						    // C is container of candidate neighbors nodes
				double d = distance(q, nodes[c].data);
				W.emplace(d,c);
				inW.insert(c);
			}
			
			
			if(extendCandidates){ // if extendCandidates is true ...
				
				for(int e : C){ // ... then for each e in C
					
					for(int eNeighbor : nodes[e].neighbors[lc]){ // ... put all of e's neighbors ... 
						
						if(!inW.count(eNeighbor)){
							
							double d2 = distance(q, nodes[eNeighbor].data);
							W.emplace(d2,eNeighbor); // ... into W, the working set
							inW.insert(eNeighbor);
						}
					}
				}
		   }
		   
		   
		   while(!W.empty() && (int)R.size() < M){ // keep looping while there's still nodes in the working set and q's has less than M outneighbors
			   
			   // out of all of W's items, extract the one that's closest to q
			   auto [d_e, e] = W.top(); // d_e is the distance between q and e
										// e is the index of the node
			   W.pop();
			   
			   // if e is closer to q compared to ANY r in R ...
			   bool eCanEnterR = false;
			   for(int r : R){
				   
				   if(d_e < distance(q, nodes[r].data)){
					   eCanEnterR = true;
					   break;
			       }
			   }
			   
			   // ... then put e into R
			   if(eCanEnterR){ R.push_back(e);}
			   else{ Wd.emplace(d_e, e);}
		  }
		  
		  
		  if(keepPrunedConnections){ // if keepPrunedConnections is activated ...
			  
			  while(!Wd.empty() && (int)R.size() < M){
				  
				  R.push_back(Wd.top().second); // ... put the best pruned nodes into our results container
				  Wd.pop();
			  }
		 }
					
				
		return R; // return our container of confirmed neighbors of q
			
		} // end of selectNeighborsHeuristic method definition
		
		int getNearest(const Point &q, const vector<int>& W){ // getNearest method definition
															  // out of all the elements in W, grab the one that's closest to q
									
			int best = W.front();
			double bestDistance = distance(q,nodes[best].data);
			for(int v : W){
				double d = distance(q, nodes[v].data);
				if(d < bestDistance){
					bestDistance = d;
					best = v;
				}
			}
			return best; // return an INDEX to the best point. NOT the best point itself
			
		} // end of getNearest method definition
			

}; // end of class HNSW definition

/* =========================
 * main()
 * where all of the action occurs
 * ===========================
*/
int main(int argc, char **argv)
{
	
	vector<vector<float>> dataSet = load_fvecs("siftSmall/siftsmall_base.fvecs"); // load the entire dataset into dataSet
	
	HNSW myHNSW = HNSW(16,16,32,200,1.0);  // initialize an empty HNSW index
										   // for instance...
											   // I want my HNSW index, myHNSW to have
											   // - target out degree of 16
											   // - limit of 16 neighbors for nodes on layers above ground layer
											   // - limit of 32 neighbors for nodes on the ground layer
											   // - beam width of 200 on the construction phase
											   // - normalization value of 1.0
	
	for(int i = 0; i < dataSet.size() ; ++i){ // iterate through the entire dataset and insert each point into the HNSW index
		Point currentPoint = dataSet[i];
		myHNSW.insert(currentPoint);	
	}
	
	vector<vector<float>> querySet = load_fvecs("siftSmall/siftsmall_query.fvecs"); // load the query set into querySet
	
	/* 
	 * 
	 * Below is testing to see if HNSW's Search works... 
	 * 
	*/
	// for example, let's say I want to find the closest neighbor(s) to the FOURTH query in the query set
	Point myFourthQuery = querySet[3]; // grab my FOURTH query point ---- q
	cout << "My query under investigation looks like: " << myFourthQuery << "\n";
	vector<int> containerOfIndicesOf_K_ClosestDataPointsTo_myFourthQuery = myHNSW.searchKNN(myFourthQuery,1,8); // run the HNSW search
																												// the search will return a container of k-many node indices representing which nodes are the closest to the query q
																												// - for this example, I set k to be 1 ---> I want the SINGLE closest to q
																												// - for this example, I set ef to be 8 ---> I want my search beam width to be 8
																												
	cout << "The single closest data point to myFourthQuery is the " << containerOfIndicesOf_K_ClosestDataPointsTo_myFourthQuery[0] << "th data point \n";
	cout << "And that data point looks like: " << myHNSW.pointGetter(containerOfIndicesOf_K_ClosestDataPointsTo_myFourthQuery[0]);
	
	return 0;
} // end of main()








