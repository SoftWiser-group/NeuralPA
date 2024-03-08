[BugLab_Variable_Misuse]^this.primitiveType = primitiveType;^71^^^^^68^74^this.primitiveType = elementType;^[CLASS] EnumElementType  [METHOD] <init> [RETURN_TYPE] String)   JSTypeRegistry registry JSType elementType String name [VARIABLES] ObjectType  primitiveObjectType  JSTypeRegistry  registry  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^this.primitiveObjectType = primitiveType.toObjectType (  ) ;^72^^^^^68^74^this.primitiveObjectType = elementType.toObjectType (  ) ;^[CLASS] EnumElementType  [METHOD] <init> [RETURN_TYPE] String)   JSTypeRegistry registry JSType elementType String name [VARIABLES] ObjectType  primitiveObjectType  JSTypeRegistry  registry  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^78^^^^^77^79^return true;^[CLASS] EnumElementType  [METHOD] isEnumElementType [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return elementType.matchesNumberContext (  ) ;^83^^^^^82^84^return primitiveType.matchesNumberContext (  ) ;^[CLASS] EnumElementType  [METHOD] matchesNumberContext [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return elementType.matchesStringContext (  ) ;^88^^^^^87^89^return primitiveType.matchesStringContext (  ) ;^[CLASS] EnumElementType  [METHOD] matchesStringContext [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return elementType.matchesObjectContext (  ) ;^93^^^^^92^94^return primitiveType.matchesObjectContext (  ) ;^[CLASS] EnumElementType  [METHOD] matchesObjectContext [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return elementType.canBeCalled (  ) ;^98^^^^^97^99^return primitiveType.canBeCalled (  ) ;^[CLASS] EnumElementType  [METHOD] canBeCalled [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return elementType.isObject (  ) ;^103^^^^^102^104^return primitiveType.isObject (  ) ;^[CLASS] EnumElementType  [METHOD] isObject [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return primitiveType.testForEquality ( elementType ) ;^108^^^^^107^109^return primitiveType.testForEquality ( that ) ;^[CLASS] EnumElementType  [METHOD] testForEquality [RETURN_TYPE] TernaryValue   JSType that [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return elementType.testForEquality ( that ) ;^108^^^^^107^109^return primitiveType.testForEquality ( that ) ;^[CLASS] EnumElementType  [METHOD] testForEquality [RETURN_TYPE] TernaryValue   JSType that [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return that.testForEquality ( primitiveType ) ;^108^^^^^107^109^return primitiveType.testForEquality ( that ) ;^[CLASS] EnumElementType  [METHOD] testForEquality [RETURN_TYPE] TernaryValue   JSType that [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return that.isNullable (  ) ;^118^^^^^117^119^return primitiveType.isNullable (  ) ;^[CLASS] EnumElementType  [METHOD] isNullable [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( this <= that )  {^128^^^^^127^137^if  ( this == that )  {^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^} else if  ( that instanceof JSType || this.isNominalType (  )  )  {^130^^^^^127^137^} else if  ( that instanceof JSType && this.isNominalType (  )  )  {^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^} else if  ( that  >>  JSType && this.isNominalType (  )  )  {^130^^^^^127^137^} else if  ( that instanceof JSType && this.isNominalType (  )  )  {^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( primitiveObjectType != null && thatObj.isNominalType (  )  )  {^132^^^^^127^137^if  ( thatObj != null && thatObj.isNominalType (  )  )  {^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( thatObj != null || thatObj.isNominalType (  )  )  {^132^^^^^127^137^if  ( thatObj != null && thatObj.isNominalType (  )  )  {^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( thatObj == null && thatObj.isNominalType (  )  )  {^132^^^^^127^137^if  ( thatObj != null && thatObj.isNominalType (  )  )  {^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return getReferenceName (  ) .equals ( primitiveObjectType.getReferenceName (  )  ) ;^133^^^^^127^137^return getReferenceName (  ) .equals ( thatObj.getReferenceName (  )  ) ;^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^129^^^^^127^137^return true;^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^} else if  ( that  &&  JSType && this.isNominalType (  )  )  {^130^^^^^127^137^} else if  ( that instanceof JSType && this.isNominalType (  )  )  {^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return true;^136^^^^^127^137^return false;^[CLASS] EnumElementType  [METHOD] equals [RETURN_TYPE] boolean   Object that [VARIABLES] ObjectType  primitiveObjectType  thatObj  Object  that  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return getReferenceName (  )  + ".<" + that + ">";^154^^^^^153^155^return getReferenceName (  )  + ".<" + primitiveType + ">";^[CLASS] EnumElementType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return getReferenceName (  <=  )  + ".<" + primitiveType + ">";^154^^^^^153^155^return getReferenceName (  )  + ".<" + primitiveType + ">";^[CLASS] EnumElementType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return getReferenceName (  <<  )  + ".<" + primitiveType + ">";^154^^^^^153^155^return getReferenceName (  )  + ".<" + primitiveType + ">";^[CLASS] EnumElementType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return getReferenceName (  )   >=  ".<" + primitiveType + ">";^154^^^^^153^155^return getReferenceName (  )  + ".<" + primitiveType + ">";^[CLASS] EnumElementType  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^164^^^^^163^165^return true;^[CLASS] EnumElementType  [METHOD] hasReferenceName [RETURN_TYPE] boolean   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( JSType.isSubtype ( this, elementType )  )  {^169^^^^^168^174^if  ( JSType.isSubtype ( this, that )  )  {^[CLASS] EnumElementType  [METHOD] isSubtype [RETURN_TYPE] boolean   JSType that [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return elementType.isSubtype ( that ) ;^172^^^^^168^174^return primitiveType.isSubtype ( that ) ;^[CLASS] EnumElementType  [METHOD] isSubtype [RETURN_TYPE] boolean   JSType that [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return that.isSubtype ( primitiveType ) ;^172^^^^^168^174^return primitiveType.isSubtype ( that ) ;^[CLASS] EnumElementType  [METHOD] isSubtype [RETURN_TYPE] boolean   JSType that [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return primitiveType.isSubtype ( elementType ) ;^172^^^^^168^174^return primitiveType.isSubtype ( that ) ;^[CLASS] EnumElementType  [METHOD] isSubtype [RETURN_TYPE] boolean   JSType that [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^170^^^^^168^174^return true;^[CLASS] EnumElementType  [METHOD] isSubtype [RETURN_TYPE] boolean   JSType that [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( JSType.isSubtype ( this, primitiveType )  )  {^169^^^^^168^174^if  ( JSType.isSubtype ( this, that )  )  {^[CLASS] EnumElementType  [METHOD] isSubtype [RETURN_TYPE] boolean   JSType that [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return null.caseEnumElementType ( this ) ;^178^^^^^177^179^return visitor.caseEnumElementType ( this ) ;^[CLASS] EnumElementType  [METHOD] visit [RETURN_TYPE] <T>   Visitor<T> visitor [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  String  name  boolean  Visitor  visitor  long  serialVersionUID  
[BugLab_Wrong_Literal]^return false;^185^^^^^182^186^return true;^[CLASS] EnumElementType  [METHOD] defineProperty [RETURN_TYPE] boolean   String propertyName JSType type boolean inferred boolean inExterns [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  inExterns  inferred  long  serialVersionUID  
[BugLab_Variable_Misuse]^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeDeclared ( name ) ;^190^191^^^^189^192^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return thatObj == null ? false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^190^191^^^^189^192^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return propertyName == null ? false : primitiveObjectType.isPropertyTypeDeclared ( primitiveObjectType ) ;^190^191^^^^189^192^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return primitiveObjectType != null ? false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^190^191^^^^189^192^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return primitiveObjectType == null ? true : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^190^191^^^^189^192^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^false : primitiveObjectType.isPropertyTypeDeclared ( name ) ;^191^^^^^189^192^false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^false : thatObj.isPropertyTypeDeclared ( propertyName ) ;^191^^^^^189^192^false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^false : propertyName.isPropertyTypeDeclared ( primitiveObjectType ) ;^191^^^^^189^192^false : primitiveObjectType.isPropertyTypeDeclared ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeDeclared [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeInferred ( name ) ;^196^197^^^^195^198^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeInferred [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return thatObj == null ? false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^196^197^^^^195^198^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeInferred [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return propertyName == null ? false : primitiveObjectType.isPropertyTypeInferred ( primitiveObjectType ) ;^196^197^^^^195^198^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeInferred [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return primitiveObjectType != null ? false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^196^197^^^^195^198^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeInferred [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return primitiveObjectType == null ? true : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^196^197^^^^195^198^return primitiveObjectType == null ? false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeInferred [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^false : thatObj.isPropertyTypeInferred ( propertyName ) ;^197^^^^^195^198^false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeInferred [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^false : propertyName.isPropertyTypeInferred ( primitiveObjectType ) ;^197^^^^^195^198^false : primitiveObjectType.isPropertyTypeInferred ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] isPropertyTypeInferred [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return thatObj == null ? 0 : primitiveObjectType.getPropertiesCount (  ) ;^207^208^^^^206^209^return primitiveObjectType == null ? 0 : primitiveObjectType.getPropertiesCount (  ) ;^[CLASS] EnumElementType  [METHOD] getPropertiesCount [RETURN_TYPE] int   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return primitiveObjectType != null ? 0 : primitiveObjectType.getPropertiesCount (  ) ;^207^208^^^^206^209^return primitiveObjectType == null ? 0 : primitiveObjectType.getPropertiesCount (  ) ;^[CLASS] EnumElementType  [METHOD] getPropertiesCount [RETURN_TYPE] int   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^if  ( thatObj != null )  {^213^^^^^212^216^if  ( primitiveObjectType != null )  {^[CLASS] EnumElementType  [METHOD] collectPropertyNames [RETURN_TYPE] void   String> props [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  Set  props  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^if  ( primitiveObjectType == null )  {^213^^^^^212^216^if  ( primitiveObjectType != null )  {^[CLASS] EnumElementType  [METHOD] collectPropertyNames [RETURN_TYPE] void   String> props [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  Set  props  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return primitiveType.findPropertyType ( name ) ;^220^^^^^219^221^return primitiveType.findPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] findPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return propertyName.findPropertyType ( primitiveType ) ;^220^^^^^219^221^return primitiveType.findPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] findPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return type.findPropertyType ( propertyName ) ;^220^^^^^219^221^return primitiveType.findPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] findPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return primitiveObjectType == null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( name ) ;^225^226^227^^^224^228^return primitiveObjectType == null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] getPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return thatObj == null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( propertyName ) ;^225^226^227^^^224^228^return primitiveObjectType == null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] getPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return propertyName == null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( primitiveObjectType ) ;^225^226^227^^^224^228^return primitiveObjectType == null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] getPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return primitiveObjectType != null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( propertyName ) ;^225^226^227^^^224^228^return primitiveObjectType == null ? getNativeType ( JSTypeNative.UNKNOWN_TYPE )  : primitiveObjectType.getPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] getPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^primitiveObjectType.getPropertyType ( name ) ;^227^^^^^224^228^primitiveObjectType.getPropertyType ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] getPropertyType [RETURN_TYPE] JSType   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return primitiveObjectType == null ? false : primitiveObjectType.hasProperty ( name ) ;^232^233^234^^^231^235^return primitiveObjectType == null ? false : primitiveObjectType.hasProperty ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] hasProperty [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return propertyName == null ? false : primitiveObjectType.hasProperty ( primitiveObjectType ) ;^232^233^234^^^231^235^return primitiveObjectType == null ? false : primitiveObjectType.hasProperty ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] hasProperty [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return primitiveObjectType != null ? false : primitiveObjectType.hasProperty ( propertyName ) ;^232^233^234^^^231^235^return primitiveObjectType == null ? false : primitiveObjectType.hasProperty ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] hasProperty [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Literal]^return primitiveObjectType == null ? true : primitiveObjectType.hasProperty ( propertyName ) ;^232^233^234^^^231^235^return primitiveObjectType == null ? false : primitiveObjectType.hasProperty ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] hasProperty [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^primitiveObjectType.hasProperty ( name ) ;^234^^^^^231^235^primitiveObjectType.hasProperty ( propertyName ) ;^[CLASS] EnumElementType  [METHOD] hasProperty [RETURN_TYPE] boolean   String propertyName [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return thatObj == null ? null : primitiveObjectType.getConstructor (  ) ;^239^240^^^^238^241^return primitiveObjectType == null ? null : primitiveObjectType.getConstructor (  ) ;^[CLASS] EnumElementType  [METHOD] getConstructor [RETURN_TYPE] FunctionType   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return primitiveObjectType != null ? null : primitiveObjectType.getConstructor (  ) ;^239^240^^^^238^241^return primitiveObjectType == null ? null : primitiveObjectType.getConstructor (  ) ;^[CLASS] EnumElementType  [METHOD] getConstructor [RETURN_TYPE] FunctionType   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return type.autoboxesTo (  ) ;^245^^^^^244^246^return primitiveType.autoboxesTo (  ) ;^[CLASS] EnumElementType  [METHOD] autoboxesTo [RETURN_TYPE] JSType   [VARIABLES] ObjectType  primitiveObjectType  thatObj  JSType  elementType  primitiveType  that  type  String  name  propertyName  boolean  long  serialVersionUID  