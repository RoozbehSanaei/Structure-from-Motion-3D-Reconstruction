#pragma once

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minijson {

struct Value {
  enum class Type { Null, Bool, Number, String, Object, Array };
  Type type = Type::Null;

  bool b = false;
  double num = 0.0;
  std::string str;
  std::unordered_map<std::string, Value> obj;
  std::vector<Value> arr;

  static Value make_null(){ return Value{}; }
  static Value make_bool(bool v){ Value x; x.type = Type::Bool; x.b = v; return x; }
  static Value make_number(double v){ Value x; x.type = Type::Number; x.num = v; return x; }
  static Value make_string(std::string v){ Value x; x.type = Type::String; x.str = std::move(v); return x; }
  static Value make_object(){ Value x; x.type = Type::Object; return x; }
  static Value make_array(){ Value x; x.type = Type::Array; return x; }

  bool is_null()   const { return type == Type::Null; }
  bool is_bool()   const { return type == Type::Bool; }
  bool is_number() const { return type == Type::Number; }
  bool is_string() const { return type == Type::String; }
  bool is_object() const { return type == Type::Object; }
  bool is_array()  const { return type == Type::Array; }

  const Value* get(const std::string& key) const {
    if(!is_object()) return nullptr;
    auto it = obj.find(key);
    if(it == obj.end()) return nullptr;
    return &it->second;
  }
};

class Parser {
public:
  explicit Parser(const std::string& s): p_(s.c_str()), end_(s.c_str() + s.size()) {}

  Value parse(){
    ws_();
    Value v = parse_value_();
    ws_();
    if(p_ != end_) throw std::runtime_error("Trailing characters after JSON");
    return v;
  }

private:
  const char* p_;
  const char* end_;

  void ws_(){
    while(p_ < end_ && std::isspace((unsigned char)*p_)) ++p_;
  }

  [[noreturn]] void err_(const std::string& msg){
    throw std::runtime_error("JSON parse error: " + msg);
  }

  bool match_(char c){
    if(p_ < end_ && *p_ == c){ ++p_; return true; }
    return false;
  }

  void expect_(char c){
    if(!match_(c)){
      std::string m = "Expected '";
      m.push_back(c);
      m += "'";
      err_(m);
    }
  }

  Value parse_value_(){
    ws_();
    if(p_ >= end_) err_("Unexpected end of input");
    const char c = *p_;
    if(c == 'n') return parse_null_();
    if(c == 't' || c == 'f') return parse_bool_();
    if(c == '"') return Value::make_string(parse_string_());
    if(c == '{') return parse_object_();
    if(c == '[') return parse_array_();
    if(c == '-' || std::isdigit((unsigned char)c)) return Value::make_number(parse_number_());
    err_(std::string("Unexpected character '") + c + "'");
    return Value::make_null();
  }

  Value parse_null_(){
    if(end_ - p_ >= 4 && p_[0]=='n' && p_[1]=='u' && p_[2]=='l' && p_[3]=='l'){
      p_ += 4;
      return Value::make_null();
    }
    err_("Invalid token (expected null)");
    return Value::make_null();
  }

  Value parse_bool_(){
    if(end_ - p_ >= 4 && p_[0]=='t' && p_[1]=='r' && p_[2]=='u' && p_[3]=='e'){
      p_ += 4;
      return Value::make_bool(true);
    }
    if(end_ - p_ >= 5 && p_[0]=='f' && p_[1]=='a' && p_[2]=='l' && p_[3]=='s' && p_[4]=='e'){
      p_ += 5;
      return Value::make_bool(false);
    }
    err_("Invalid token (expected true/false)");
    return Value::make_bool(false);
  }

  static int hex_(char c){
    if(c >= '0' && c <= '9') return c - '0';
    if(c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if(c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
  }

  std::string parse_string_(){
    expect_('"');
    std::string out;
    while(p_ < end_){
      char c = *p_++;
      if(c == '"') return out;
      if(c == '\\'){
        if(p_ >= end_) err_("Bad escape");
        char e = *p_++;
        switch(e){
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            // minimal \uXXXX handling; only BMP and without surrogate pair expansion
            if(end_ - p_ < 4) err_("Bad \\u escape");
            int v = 0;
            for(int i=0;i<4;i++){
              int h = hex_(p_[i]);
              if(h < 0) err_("Bad hex in \\u escape");
              v = (v<<4) | h;
            }
            p_ += 4;
            if(v <= 0x7F) out.push_back((char)v);
            else if(v <= 0x7FF){
              out.push_back((char)(0xC0 | ((v>>6)&0x1F)));
              out.push_back((char)(0x80 | (v&0x3F)));
            } else {
              out.push_back((char)(0xE0 | ((v>>12)&0x0F)));
              out.push_back((char)(0x80 | ((v>>6)&0x3F)));
              out.push_back((char)(0x80 | (v&0x3F)));
            }
            break;
          }
          default: err_("Unknown escape");
        }
      } else {
        out.push_back(c);
      }
    }
    err_("Unterminated string");
    return out;
  }

  double parse_number_(){
    const char* start = p_;
    if(match_('-')){}
    if(p_ >= end_) err_("Bad number");
    if(*p_ == '0'){
      ++p_;
    } else {
      if(!std::isdigit((unsigned char)*p_)) err_("Bad number");
      while(p_ < end_ && std::isdigit((unsigned char)*p_)) ++p_;
    }
    if(p_ < end_ && *p_ == '.'){
      ++p_;
      if(p_ >= end_ || !std::isdigit((unsigned char)*p_)) err_("Bad fraction");
      while(p_ < end_ && std::isdigit((unsigned char)*p_)) ++p_;
    }
    if(p_ < end_ && (*p_ == 'e' || *p_ == 'E')){
      ++p_;
      if(p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
      if(p_ >= end_ || !std::isdigit((unsigned char)*p_)) err_("Bad exponent");
      while(p_ < end_ && std::isdigit((unsigned char)*p_)) ++p_;
    }
    std::string tmp(start, p_);
    char* ep = nullptr;
    const double v = std::strtod(tmp.c_str(), &ep);
    if(ep == tmp.c_str()) err_("Bad number conversion");
    return v;
  }

  Value parse_array_(){
    expect_('[');
    Value out = Value::make_array();
    ws_();
    if(match_(']')) return out;
    while(true){
      out.arr.push_back(parse_value_());
      ws_();
      if(match_(']')) break;
      expect_(',');
      ws_();
    }
    return out;
  }

  Value parse_object_(){
    expect_('{');
    Value out = Value::make_object();
    ws_();
    if(match_('}')) return out;
    while(true){
      if(p_ >= end_ || *p_ != '"') err_("Expected string key");
      std::string key = parse_string_();
      ws_();
      expect_(':');
      ws_();
      Value v = parse_value_();
      out.obj.emplace(std::move(key), std::move(v));
      ws_();
      if(match_('}')) break;
      expect_(',');
      ws_();
    }
    return out;
  }
};

inline Value parse(const std::string& s){
  Parser p(s);
  return p.parse();
}

} // namespace minijson
