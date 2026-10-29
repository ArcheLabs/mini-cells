// Exact stride-1 fragment audit. Rolling hash is only a filter, never match evidence.
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>
struct Pattern { std::vector<uint32_t> v; uint64_t count=0, budget_count=0; std::array<uint64_t,3> sample{}; };
static constexpr uint64_t B=1000003;
uint32_t u32(std::ifstream &f) { uint32_t v; f.read(reinterpret_cast<char*>(&v),4); if(!f) throw std::runtime_error("TRUNCATED_PATTERN"); return v; }
uint64_t hash(const std::vector<uint32_t> &v) { uint64_t h=0; for(auto x:v) h=h*B+uint64_t(x)+1; return h; }
int main(int argc,char**argv) {
 try {
  if(argc!=5) throw std::runtime_error("ARGS_PATTERNS_SOURCE_OUTPUT_BUDGET");
  std::ifstream pf(argv[1],std::ios::binary); char magic[4]; pf.read(magic,4);
  if(!pf || std::string(magic,4)!="MCF1") throw std::runtime_error("PATTERN_MAGIC");
  auto n=u32(pf); std::vector<Pattern> patterns(n);
  std::map<uint32_t,std::unordered_map<uint64_t,std::vector<uint32_t>>> indices;
  for(uint32_t i=0;i<n;i++) { auto len=u32(pf); if(len==0 || len>1024) throw std::runtime_error("INVALID_PATTERN_LENGTH");
   patterns[i].v.resize(len); pf.read(reinterpret_cast<char*>(patterns[i].v.data()),uint64_t(len)*4);
   if(!pf) throw std::runtime_error("TRUNCATED_PATTERN"); indices[len][hash(patterns[i].v)].push_back(i); }
  std::ifstream sf(argv[2],std::ios::binary|std::ios::ate); if(!sf) throw std::runtime_error("SOURCE_MISSING");
  uint64_t bytes=sf.tellg(); if(bytes%4) throw std::runtime_error("SOURCE_ALIGNMENT");
  std::vector<uint32_t> source(bytes/4); sf.seekg(0); sf.read(reinterpret_cast<char*>(source.data()),bytes);
  if(!sf && bytes) throw std::runtime_error("SOURCE_READ");
  uint64_t budget=std::stoull(argv[4]); if(budget>source.size()) throw std::runtime_error("BUDGET_OUT_OF_BOUNDS");
  for(auto &pair:indices) { auto len=pair.first; auto &lookup=pair.second; if(source.size()<len) continue;
   uint64_t h=0,power=1; for(uint32_t i=0;i<len;i++) { h=h*B+uint64_t(source[i])+1; if(i+1<len) power*=B; }
   for(uint64_t start=0;start+len<=source.size();start++) {
    auto it=lookup.find(h); if(it!=lookup.end()) for(auto id:it->second) {
     auto &p=patterns[id]; bool same=true; for(uint32_t j=0;j<len;j++) if(source[start+j]!=p.v[j]) {same=false;break;}
     if(same) { if(p.count<3) p.sample[p.count]=start; ++p.count; if(start+len<=budget) ++p.budget_count; }
    }
    if(start+len<source.size()) h=(h-(uint64_t(source[start])+1)*power)*B+uint64_t(source[start+len])+1;
   }
  }
  std::ofstream out(argv[3]); if(!out) throw std::runtime_error("OUTPUT_OPEN");
  for(uint32_t i=0;i<n;i++) {auto &p=patterns[i]; if(p.count) { out<<i<<'\t'<<p.v.size()<<'\t'<<p.count<<'\t'<<p.budget_count; for(uint64_t j=0;j<3 && j<p.count;j++) out<<'\t'<<p.sample[j]; out<<'\n'; }}
  if(!out) throw std::runtime_error("OUTPUT_WRITE");
  return 0;
 } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
