[BugLab_Argument_Swapping]^return put ( content, paramName, content.getClass (  )  ) ;^37^^^^^36^38^return put ( paramName, content, content.getClass (  )  ) ;^[CLASS] ParamMap Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content [VARIABLES] Object  content  String  paramName  boolean  T  spec  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( spec.checkIfCompatible ( typeOfContent, paramName )  ) ;^41^^^^^40^44^Preconditions.checkArgument ( spec.checkIfCompatible ( paramName, typeOfContent )  ) ;^[CLASS] ParamMap Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content Type typeOfContent [VARIABLES] Type  typeOfContent  boolean  T  spec  Object  content  String  paramName  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( typeOfContent.checkIfCompatible ( paramName, spec )  ) ;^41^^^^^40^44^Preconditions.checkArgument ( spec.checkIfCompatible ( paramName, typeOfContent )  ) ;^[CLASS] ParamMap Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content Type typeOfContent [VARIABLES] Type  typeOfContent  boolean  T  spec  Object  content  String  paramName  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( paramName.checkIfCompatible ( spec, typeOfContent )  ) ;^41^^^^^40^44^Preconditions.checkArgument ( spec.checkIfCompatible ( paramName, typeOfContent )  ) ;^[CLASS] ParamMap Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content Type typeOfContent [VARIABLES] Type  typeOfContent  boolean  T  spec  Object  content  String  paramName  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^contents.put ( param.getName (  ) , paramKey ) ;^47^^^^^46^49^contents.put ( paramKey.getName (  ) , param ) ;^[CLASS] ParamMap Builder  [METHOD] put [RETURN_TYPE] <K>   TypedKey<K> paramKey K param [VARIABLES] K  param  boolean  T  spec  TypedKey  paramKey  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^contents.put ( paramKeyKey.getName (  ) , param ) ;^47^^^^^46^49^contents.put ( paramKey.getName (  ) , param ) ;^[CLASS] ParamMap Builder  [METHOD] put [RETURN_TYPE] <K>   TypedKey<K> paramKey K param [VARIABLES] K  param  boolean  T  spec  TypedKey  paramKey  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^return paramName.get ( contents ) ;^65^^^^^64^66^return contents.get ( paramName ) ;^[CLASS] ParamMap Builder  [METHOD] get [RETURN_TYPE] Object   String paramName [VARIABLES] String  paramName  boolean  T  spec  Map  contents  ParamMapSpec  spec  
[BugLab_Variable_Misuse]^return  ( T )  get ( 1.getName (  ) , key.getClassOfT (  )  ) ;^70^^^^^69^71^return  ( T )  get ( key.getName (  ) , key.getClassOfT (  )  ) ;^[CLASS] ParamMap Builder  [METHOD] get [RETURN_TYPE] <T>   TypedKey<T> key [VARIABLES] boolean  T  spec  TypedKey  key  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( key.checkIfCompatible ( spec, typeOfValue ) , "Incompatible key %s for type %s", key, typeOfValue ) ;^75^76^^^^74^78^Preconditions.checkArgument ( spec.checkIfCompatible ( key, typeOfValue ) , "Incompatible key %s for type %s", key, typeOfValue ) ;^[CLASS] ParamMap Builder  [METHOD] get [RETURN_TYPE] <T>   String key Type typeOfValue [VARIABLES] Type  typeOfValue  String  key  boolean  T  spec  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( typeOfValue.checkIfCompatible ( key, spec ) , "Incompatible key %s for type %s", key, typeOfValue ) ;^75^76^^^^74^78^Preconditions.checkArgument ( spec.checkIfCompatible ( key, typeOfValue ) , "Incompatible key %s for type %s", key, typeOfValue ) ;^[CLASS] ParamMap Builder  [METHOD] get [RETURN_TYPE] <T>   String key Type typeOfValue [VARIABLES] Type  typeOfValue  String  key  boolean  T  spec  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( spec.checkIfCompatible ( typeOfValue, key ) , "Incompatible key %s for type %s", key, typeOfValue ) ;^75^76^^^^74^78^Preconditions.checkArgument ( spec.checkIfCompatible ( key, typeOfValue ) , "Incompatible key %s for type %s", key, typeOfValue ) ;^[CLASS] ParamMap Builder  [METHOD] get [RETURN_TYPE] <T>   String key Type typeOfValue [VARIABLES] Type  typeOfValue  String  key  boolean  T  spec  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^return  ( T )  key.get ( contents ) ;^77^^^^^74^78^return  ( T )  contents.get ( key ) ;^[CLASS] ParamMap Builder  [METHOD] get [RETURN_TYPE] <T>   String key Type typeOfValue [VARIABLES] Type  typeOfValue  String  key  boolean  T  spec  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^return headerName.getTypeFor ( spec ) ;^81^^^^^80^82^return spec.getTypeFor ( headerName ) ;^[CLASS] ParamMap Builder  [METHOD] getSpec [RETURN_TYPE] Type   String headerName [VARIABLES] String  headerName  boolean  T  spec  Map  contents  ParamMapSpec  spec  
[BugLab_Argument_Swapping]^return put ( content, paramName, content.getClass (  )  ) ;^37^^^^^36^38^return put ( paramName, content, content.getClass (  )  ) ;^[CLASS] Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content [VARIABLES] Object  content  String  paramName  boolean  T  spec  Map  contents  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( spec.checkIfCompatible ( typeOfContent, paramName )  ) ;^41^^^^^40^44^Preconditions.checkArgument ( spec.checkIfCompatible ( paramName, typeOfContent )  ) ;^[CLASS] Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content Type typeOfContent [VARIABLES] Object  content  Type  typeOfContent  String  paramName  boolean  T  spec  Map  contents  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( typeOfContent.checkIfCompatible ( paramName, spec )  ) ;^41^^^^^40^44^Preconditions.checkArgument ( spec.checkIfCompatible ( paramName, typeOfContent )  ) ;^[CLASS] Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content Type typeOfContent [VARIABLES] Object  content  Type  typeOfContent  String  paramName  boolean  T  spec  Map  contents  
[BugLab_Argument_Swapping]^Preconditions.checkArgument ( paramName.checkIfCompatible ( spec, typeOfContent )  ) ;^41^^^^^40^44^Preconditions.checkArgument ( spec.checkIfCompatible ( paramName, typeOfContent )  ) ;^[CLASS] Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content Type typeOfContent [VARIABLES] Object  content  Type  typeOfContent  String  paramName  boolean  T  spec  Map  contents  
[BugLab_Argument_Swapping]^contents.put ( content, paramName ) ;^42^^^^^40^44^contents.put ( paramName, content ) ;^[CLASS] Builder  [METHOD] put [RETURN_TYPE] ParamMap$Builder   String paramName Object content Type typeOfContent [VARIABLES] Object  content  Type  typeOfContent  String  paramName  boolean  T  spec  Map  contents  
[BugLab_Argument_Swapping]^contents.put ( paramKeyKey.getName (  ) , param ) ;^47^^^^^46^49^contents.put ( paramKey.getName (  ) , param ) ;^[CLASS] Builder  [METHOD] put [RETURN_TYPE] <K>   TypedKey<K> paramKey K param [VARIABLES] K  param  boolean  T  spec  TypedKey  paramKey  Map  contents  