[BugLab_Argument_Swapping]^super ( contents, spec ) ;^55^^^^^54^56^super ( spec, contents ) ;^[CLASS] HeaderMap Builder  [METHOD] <init> [RETURN_TYPE] Map)   HeaderMapSpec spec Object> contents [VARIABLES] boolean  Map  contents  HeaderMapSpec  spec  
[BugLab_Argument_Swapping]^return put ( content.getName (  ) , paramKey ) ;^36^^^^^35^37^return put ( paramKey.getName (  ) , content ) ;^[CLASS] HeaderMap Builder  [METHOD] put [RETURN_TYPE] <T>   TypedKey<T> paramKey T content [VARIABLES] boolean  TypedKey  paramKey  T  content  
[BugLab_Argument_Swapping]^return new HeaderMap ( contents, spec ) ;^50^^^^^49^51^return new HeaderMap ( spec, contents ) ;^[CLASS] HeaderMap Builder  [METHOD] build [RETURN_TYPE] HeaderMap   [VARIABLES] boolean  
[BugLab_Argument_Swapping]^return put ( content.getName (  ) , paramKey ) ;^36^^^^^35^37^return put ( paramKey.getName (  ) , content ) ;^[CLASS] Builder  [METHOD] put [RETURN_TYPE] <T>   TypedKey<T> paramKey T content [VARIABLES] boolean  TypedKey  paramKey  T  content  
[BugLab_Argument_Swapping]^return new HeaderMap ( contents, spec ) ;^50^^^^^49^51^return new HeaderMap ( spec, contents ) ;^[CLASS] Builder  [METHOD] build [RETURN_TYPE] HeaderMap   [VARIABLES] boolean  
[BugLab_Variable_Misuse]^return new HeaderMap ( spec, null ) ;^50^^^^^49^51^return new HeaderMap ( spec, contents ) ;^[CLASS] Builder  [METHOD] build [RETURN_TYPE] HeaderMap   [VARIABLES] boolean  
