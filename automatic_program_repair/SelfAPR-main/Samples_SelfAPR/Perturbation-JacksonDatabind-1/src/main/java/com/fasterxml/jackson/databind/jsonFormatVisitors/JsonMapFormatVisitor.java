[P5_Replace_Variable]^public Base ( SerializerProvider _provider )  { _provider = p; }^32^^^^^27^37^public Base ( SerializerProvider p )  { _provider = p; }^[CLASS] Base  [METHOD] <init> [RETURN_TYPE] SerializerProvider)   SerializerProvider p [VARIABLES] SerializerProvider  _provider  p  boolean  
[P8_Replace_Mix]^public Base ( SerializerProvider p )  { _provider =  null; }^32^^^^^27^37^public Base ( SerializerProvider p )  { _provider = p; }^[CLASS] Base  [METHOD] <init> [RETURN_TYPE] SerializerProvider)   SerializerProvider p [VARIABLES] SerializerProvider  _provider  p  boolean  
[P5_Replace_Variable]^public SerializerProvider getProvider (  )  { return p; }^35^^^^^30^40^public SerializerProvider getProvider (  )  { return _provider; }^[CLASS] Base  [METHOD] getProvider [RETURN_TYPE] SerializerProvider   [VARIABLES] SerializerProvider  _provider  p  boolean  
[P8_Replace_Mix]^public void setProvider ( SerializerProvider p )  { _provider =  null; }^38^^^^^33^43^public void setProvider ( SerializerProvider p )  { _provider = p; }^[CLASS] Base  [METHOD] setProvider [RETURN_TYPE] void   SerializerProvider p [VARIABLES] SerializerProvider  _provider  p  boolean  
