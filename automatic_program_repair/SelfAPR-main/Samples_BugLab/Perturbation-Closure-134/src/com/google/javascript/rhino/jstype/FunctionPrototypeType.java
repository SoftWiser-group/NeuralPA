[BugLab_Argument_Swapping]^this ( ownerFunction, registry, implicitPrototype, false ) ;^65^^^^^63^66^this ( registry, ownerFunction, implicitPrototype, false ) ;^[CLASS] FunctionPrototypeType  [METHOD] <init> [RETURN_TYPE] ObjectType)   JSTypeRegistry registry FunctionType ownerFunction ObjectType implicitPrototype [VARIABLES] ObjectType  implicitPrototype  JSTypeRegistry  registry  boolean  long  serialVersionUID  FunctionType  ownerFunction  
[BugLab_Argument_Swapping]^this ( registry, implicitPrototype, ownerFunction, false ) ;^65^^^^^63^66^this ( registry, ownerFunction, implicitPrototype, false ) ;^[CLASS] FunctionPrototypeType  [METHOD] <init> [RETURN_TYPE] ObjectType)   JSTypeRegistry registry FunctionType ownerFunction ObjectType implicitPrototype [VARIABLES] ObjectType  implicitPrototype  JSTypeRegistry  registry  boolean  long  serialVersionUID  FunctionType  ownerFunction  
[BugLab_Argument_Swapping]^this ( implicitPrototype, ownerFunction, registry, false ) ;^65^^^^^63^66^this ( registry, ownerFunction, implicitPrototype, false ) ;^[CLASS] FunctionPrototypeType  [METHOD] <init> [RETURN_TYPE] ObjectType)   JSTypeRegistry registry FunctionType ownerFunction ObjectType implicitPrototype [VARIABLES] ObjectType  implicitPrototype  JSTypeRegistry  registry  boolean  long  serialVersionUID  FunctionType  ownerFunction  
[BugLab_Wrong_Literal]^this ( registry, ownerFunction, implicitPrototype, true ) ;^65^^^^^63^66^this ( registry, ownerFunction, implicitPrototype, false ) ;^[CLASS] FunctionPrototypeType  [METHOD] <init> [RETURN_TYPE] ObjectType)   JSTypeRegistry registry FunctionType ownerFunction ObjectType implicitPrototype [VARIABLES] ObjectType  implicitPrototype  JSTypeRegistry  registry  boolean  long  serialVersionUID  FunctionType  ownerFunction  
[BugLab_Wrong_Operator]^if  ( ownerFunction != null )  {^70^^^^^69^75^if  ( ownerFunction == null )  {^[CLASS] FunctionPrototypeType  [METHOD] getReferenceName [RETURN_TYPE] String   [VARIABLES] long  serialVersionUID  FunctionType  ownerFunction  boolean  
[BugLab_Wrong_Operator]^return ownerFunction.getReferenceName (  ||  )  + ".prototype";^73^^^^^69^75^return ownerFunction.getReferenceName (  )  + ".prototype";^[CLASS] FunctionPrototypeType  [METHOD] getReferenceName [RETURN_TYPE] String   [VARIABLES] long  serialVersionUID  FunctionType  ownerFunction  boolean  
[BugLab_Wrong_Operator]^return ownerFunction.getReferenceName (  >  )  + ".prototype";^73^^^^^69^75^return ownerFunction.getReferenceName (  )  + ".prototype";^[CLASS] FunctionPrototypeType  [METHOD] getReferenceName [RETURN_TYPE] String   [VARIABLES] long  serialVersionUID  FunctionType  ownerFunction  boolean  
[BugLab_Wrong_Operator]^return ownerFunction != null || ownerFunction.hasReferenceName (  ) ;^79^^^^^78^80^return ownerFunction != null && ownerFunction.hasReferenceName (  ) ;^[CLASS] FunctionPrototypeType  [METHOD] hasReferenceName [RETURN_TYPE] boolean   [VARIABLES] long  serialVersionUID  FunctionType  ownerFunction  boolean  
[BugLab_Wrong_Operator]^return ownerFunction == null && ownerFunction.hasReferenceName (  ) ;^79^^^^^78^80^return ownerFunction != null && ownerFunction.hasReferenceName (  ) ;^[CLASS] FunctionPrototypeType  [METHOD] hasReferenceName [RETURN_TYPE] boolean   [VARIABLES] long  serialVersionUID  FunctionType  ownerFunction  boolean  
[BugLab_Wrong_Literal]^return false;^84^^^^^83^85^return true;^[CLASS] FunctionPrototypeType  [METHOD] isFunctionPrototypeType [RETURN_TYPE] boolean   [VARIABLES] long  serialVersionUID  FunctionType  ownerFunction  boolean  