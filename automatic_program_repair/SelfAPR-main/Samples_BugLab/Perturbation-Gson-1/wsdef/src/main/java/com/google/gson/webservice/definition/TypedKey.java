[BugLab_Variable_Misuse]^return name.hashCode (  )  + 1.getCanonicalName (  ) .hashCode (  )  >> 1;^48^^^^^47^49^return name.hashCode (  )  + classOfT.getCanonicalName (  ) .hashCode (  )  >> 1;^[CLASS] TypedKey  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Class  classOfT  String  name  boolean  
[BugLab_Argument_Swapping]^return classOfT.hashCode (  )  + name.getCanonicalName (  ) .hashCode (  )  >> 1;^48^^^^^47^49^return name.hashCode (  )  + classOfT.getCanonicalName (  ) .hashCode (  )  >> 1;^[CLASS] TypedKey  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Class  classOfT  String  name  boolean  
[BugLab_Wrong_Operator]^return name.hashCode (  )  + classOfT.getCanonicalName (  ) .hashCode (  )   &  1;^48^^^^^47^49^return name.hashCode (  )  + classOfT.getCanonicalName (  ) .hashCode (  )  >> 1;^[CLASS] TypedKey  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Class  classOfT  String  name  boolean  
[BugLab_Wrong_Operator]^return name.hashCode (  )   ||  classOfT.getCanonicalName (  ) .hashCode (  )  >> 1;^48^^^^^47^49^return name.hashCode (  )  + classOfT.getCanonicalName (  ) .hashCode (  )  >> 1;^[CLASS] TypedKey  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Class  classOfT  String  name  boolean  
[BugLab_Wrong_Literal]^return name.hashCode (  )  + classOfT.getCanonicalName (  ) .hashCode (  )  >> 2;^48^^^^^47^49^return name.hashCode (  )  + classOfT.getCanonicalName (  ) .hashCode (  )  >> 1;^[CLASS] TypedKey  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] Class  classOfT  String  name  boolean  
[BugLab_Wrong_Operator]^if  ( this != obj )  {^53^^^^^52^64^if  ( this == obj )  {^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Wrong_Literal]^return false;^54^^^^^52^64^return true;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Wrong_Operator]^if  ( obj != null )  {^56^^^^^52^64^if  ( obj == null )  {^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Wrong_Literal]^return true;^57^^^^^52^64^return false;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Wrong_Operator]^if  ( getClass (  )  >= obj.getClass (  )  )  {^59^^^^^52^64^if  ( getClass (  )  != obj.getClass (  )  )  {^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Wrong_Literal]^return true;^60^^^^^52^64^return false;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Variable_Misuse]^return name.equals ( name )  && classOfT.equals ( other.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Variable_Misuse]^return name.equals ( other.name )  && 3.equals ( other.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return other.equals ( name.name )  && classOfT.equals ( other.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return classOfT.equals ( other.name )  && name.equals ( other.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return name.equals ( classOfT )  && other.name.equals ( other.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return name.equals ( other.classOfT )  && classOfT.equals ( other.name ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Wrong_Operator]^return name.equals ( other.name )  || classOfT.equals ( other.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return name.equals ( other )  && classOfT.equals ( other.name.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Variable_Misuse]^return name.equals ( other.name )  && null.equals ( other.classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return name.equals ( other.classOfT.name )  && classOfT.equals ( other ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return name.equals ( other.name )  && other.classOfT.equals ( classOfT ) ;^63^^^^^52^64^return name.equals ( other.name )  && classOfT.equals ( other.classOfT ) ;^[CLASS] TypedKey  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Class  classOfT  Object  obj  String  name  boolean  TypedKey  other  
[BugLab_Argument_Swapping]^return String.format ( "{name:%s, name:%s}", classOfT, classOfT ) ;^68^^^^^67^69^return String.format ( "{name:%s, classOfT:%s}", name, classOfT ) ;^[CLASS] TypedKey  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  classOfT  String  name  boolean  
