#include <vector>
#include <iostream>

namespace n_at {
	
	class Dimension {
		public:
			Dimension(size_t n) : dimm_vector(n){}
			Dimension(Dimension& d) : dimm_vector(d.getsize()){
				*this = d;
			}
			~Dimension(){}
			
			void push_dimm(int d){
				dimm_vector.push_back(d);
			}
			
			int getsize(){
				return dimm_vector.size();
			}
			
			int getNumElems(){
				int total = 0;
				for(int i : dimm_vector){
					total += i;
				}
				return total;
			}
			
			friend static ostream& operator<<(ostream& os, const Dimension& d);
		
			Dimension& operator=(const Dimension& rhs){
				if(this == &rhs) return *this;
				
				this->dimm_vector = rhs.dimm_vector;
				return *this;
			}
		private:
			std::vector<int> dimm_vector;
	};
}

static ostream& operator<<(ostream& os, const Dimension& d){
	os << "(size=" << d.getsize() << "," << "elems=" << d.getNumElems << ")\n";
	for(int i : d.dimm_vector){
		os << "i=" << i << ",";
	}
	
	os << "\n";
	return os;
}
