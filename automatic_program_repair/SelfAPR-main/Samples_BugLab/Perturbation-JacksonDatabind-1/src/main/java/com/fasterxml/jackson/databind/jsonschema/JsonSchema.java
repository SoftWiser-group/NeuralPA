[BugLab_Variable_Misuse]^return schema.toString (  ) ;^59^^^^^57^60^return this.schema.toString (  ) ;^[CLASS] JsonSchema  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectNode  schema  boolean  
[BugLab_Wrong_Operator]^if  ( o < this )  return true;^71^^^^^69^80^if  ( o == this )  return true;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Wrong_Literal]^if  ( o == this )  return false;^71^^^^^69^80^if  ( o == this )  return true;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Wrong_Operator]^if  ( o != null )  return false;^72^^^^^69^80^if  ( o == null )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Wrong_Literal]^if  ( o == null )  return true;^72^^^^^69^80^if  ( o == null )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Wrong_Operator]^if  ( ! ( o  !=  JsonSchema )  )  return false;^73^^^^^69^80^if  ( ! ( o instanceof JsonSchema )  )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Wrong_Literal]^if  ( ! ( o instanceof JsonSchema )  )  return true;^73^^^^^69^80^if  ( ! ( o instanceof JsonSchema )  )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Wrong_Operator]^if  ( schema != null )  {^76^^^^^69^80^if  ( schema == null )  {^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Variable_Misuse]^return schema == null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Argument_Swapping]^return other.schema.schema == null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Argument_Swapping]^return other == null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Wrong_Operator]^return other.schema != null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Argument_Swapping]^return schema.equals ( other.schema.schema ) ;^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Argument_Swapping]^return other.schema.equals ( schema ) ;^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Variable_Misuse]^return schema.equals ( schema ) ;^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Argument_Swapping]^return other.equals ( schema.schema ) ;^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[BugLab_Variable_Misuse]^return schema;^93^^^^^87^94^return objectNode;^[CLASS] JsonSchema  [METHOD] getDefaultSchemaNode [RETURN_TYPE] JsonNode   [VARIABLES] ObjectNode  objectNode  schema  boolean  