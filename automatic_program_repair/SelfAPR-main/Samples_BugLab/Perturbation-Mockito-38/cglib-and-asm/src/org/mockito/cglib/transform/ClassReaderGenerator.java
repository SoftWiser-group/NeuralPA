[BugLab_Argument_Swapping]^this ( flags, null, r ) ;^29^^^^^28^30^this ( r, null, flags ) ;^[CLASS] ClassReaderGenerator  [METHOD] <init> [RETURN_TYPE] ClassReader,int)   ClassReader r int flags [VARIABLES] ClassReader  r  boolean  int  flags  Attribute[]  attrs  
[BugLab_Wrong_Operator]^this.attrs =  ( attrs == null )  ? attrs : new Attribute[0];^34^^^^^32^36^this.attrs =  ( attrs != null )  ? attrs : new Attribute[0];^[CLASS] ClassReaderGenerator  [METHOD] <init> [RETURN_TYPE] Attribute[],int)   ClassReader r Attribute[] attrs int flags [VARIABLES] ClassReader  r  boolean  int  flags  Attribute[]  attrs  
[BugLab_Wrong_Literal]^this.attrs =  ( attrs != null )  ? attrs : new Attribute[1];^34^^^^^32^36^this.attrs =  ( attrs != null )  ? attrs : new Attribute[0];^[CLASS] ClassReaderGenerator  [METHOD] <init> [RETURN_TYPE] Attribute[],int)   ClassReader r Attribute[] attrs int flags [VARIABLES] ClassReader  r  boolean  int  flags  Attribute[]  attrs  
[BugLab_Argument_Swapping]^r.accept ( attrs, v, flags ) ;^39^^^^^38^40^r.accept ( v, attrs, flags ) ;^[CLASS] ClassReaderGenerator  [METHOD] generateClass [RETURN_TYPE] void   ClassVisitor v [VARIABLES] ClassReader  r  ClassVisitor  v  boolean  int  flags  Attribute[]  attrs  
[BugLab_Argument_Swapping]^r.accept ( flags, attrs, v ) ;^39^^^^^38^40^r.accept ( v, attrs, flags ) ;^[CLASS] ClassReaderGenerator  [METHOD] generateClass [RETURN_TYPE] void   ClassVisitor v [VARIABLES] ClassReader  r  ClassVisitor  v  boolean  int  flags  Attribute[]  attrs  