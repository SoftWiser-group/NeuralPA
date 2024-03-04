[P8_Replace_Mix]^this.schema =  null;^38^^^^^36^39^this.schema = schema;^[CLASS] JsonSchema  [METHOD] <init> [RETURN_TYPE] ObjectNode)   ObjectNode schema [VARIABLES] ObjectNode  schema  boolean  
[P5_Replace_Variable]^return schema.toString (  ) ;^59^^^^^57^60^return this.schema.toString (  ) ;^[CLASS] JsonSchema  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectNode  schema  boolean  
[P7_Replace_Invocation]^return this.schema.hashCode (  ) ;^59^^^^^57^60^return this.schema.toString (  ) ;^[CLASS] JsonSchema  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectNode  schema  boolean  
[P8_Replace_Mix]^return schema.hashCode (  ) ;^59^^^^^57^60^return this.schema.toString (  ) ;^[CLASS] JsonSchema  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectNode  schema  boolean  
[P14_Delete_Statement]^^59^^^^^57^60^return this.schema.toString (  ) ;^[CLASS] JsonSchema  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] ObjectNode  schema  boolean  
[P7_Replace_Invocation]^return schema.toString (  ) ;^65^^^^^63^66^return schema.hashCode (  ) ;^[CLASS] JsonSchema  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] ObjectNode  schema  boolean  
[P14_Delete_Statement]^^65^^^^^63^66^return schema.hashCode (  ) ;^[CLASS] JsonSchema  [METHOD] hashCode [RETURN_TYPE] int   [VARIABLES] ObjectNode  schema  boolean  
[P2_Replace_Operator]^if  ( o != this )  return true;^71^^^^^69^80^if  ( o == this )  return true;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P3_Replace_Literal]^if  ( o == this )  return false;^71^^^^^69^80^if  ( o == this )  return true;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P2_Replace_Operator]^if  ( o != null )  return false;^72^^^^^69^80^if  ( o == null )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P3_Replace_Literal]^if  ( o == null )  return true;^72^^^^^69^80^if  ( o == null )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P8_Replace_Mix]^if  ( o == true )  return false;^72^^^^^69^80^if  ( o == null )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P2_Replace_Operator]^if  ( ! ( o  !=  JsonSchema )  )  return false;^73^^^^^69^80^if  ( ! ( o instanceof JsonSchema )  )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P3_Replace_Literal]^if  ( ! ( o instanceof JsonSchema )  )  return true;^73^^^^^69^80^if  ( ! ( o instanceof JsonSchema )  )  return false;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P15_Unwrap_Block]^return false;^73^74^75^76^77^69^80^if  ( ! ( o instanceof JsonSchema )  )  return false;  JsonSchema other =  ( JsonSchema )  o; if  ( schema == null )  { return other.schema == null; }^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P16_Remove_Block]^^73^74^75^76^77^69^80^if  ( ! ( o instanceof JsonSchema )  )  return false;  JsonSchema other =  ( JsonSchema )  o; if  ( schema == null )  { return other.schema == null; }^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P2_Replace_Operator]^if  ( schema != null )  {^76^^^^^69^80^if  ( schema == null )  {^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P6_Replace_Expression]^if  ( other.schema == null )  {^76^^^^^69^80^if  ( schema == null )  {^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P8_Replace_Mix]^if  ( schema == false )  {^76^^^^^69^80^if  ( schema == null )  {^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P15_Unwrap_Block]^return (other.schema) == null;^76^77^78^^^69^80^if  ( schema == null )  { return other.schema == null; }^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P16_Remove_Block]^^76^77^78^^^69^80^if  ( schema == null )  { return other.schema == null; }^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P2_Replace_Operator]^return other.schema != null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P5_Replace_Variable]^return schema == null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P5_Replace_Variable]^return other.schema.schema == null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P5_Replace_Variable]^return other == null;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P8_Replace_Mix]^return other.schema != null;;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P8_Replace_Mix]^return false ;^77^^^^^69^80^return other.schema == null;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P5_Replace_Variable]^return other.equals ( schema.schema ) ;^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P5_Replace_Variable]^return other.schema.equals ( schema ) ;^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P7_Replace_Invocation]^return schema .equals ( o )  ;^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P14_Delete_Statement]^^79^^^^^69^80^return schema.equals ( other.schema ) ;^[CLASS] JsonSchema  [METHOD] equals [RETURN_TYPE] boolean   Object o [VARIABLES] Object  o  boolean  ObjectNode  schema  JsonSchema  other  
[P14_Delete_Statement]^^89^90^^^^87^94^ObjectNode objectNode = JsonNodeFactory.instance.objectNode (  ) ; objectNode.put ( "type", "any" ) ;^[CLASS] JsonSchema  [METHOD] getDefaultSchemaNode [RETURN_TYPE] JsonNode   [VARIABLES] ObjectNode  objectNode  schema  boolean  
[P3_Replace_Literal]^objectNode.put ( "ty", "any" ) ;^90^^^^^87^94^objectNode.put ( "type", "any" ) ;^[CLASS] JsonSchema  [METHOD] getDefaultSchemaNode [RETURN_TYPE] JsonNode   [VARIABLES] ObjectNode  objectNode  schema  boolean  
[P3_Replace_Literal]^objectNode.put ( "type", "anyan" ) ;^90^^^^^87^94^objectNode.put ( "type", "any" ) ;^[CLASS] JsonSchema  [METHOD] getDefaultSchemaNode [RETURN_TYPE] JsonNode   [VARIABLES] ObjectNode  objectNode  schema  boolean  
[P14_Delete_Statement]^^90^^^^^87^94^objectNode.put ( "type", "any" ) ;^[CLASS] JsonSchema  [METHOD] getDefaultSchemaNode [RETURN_TYPE] JsonNode   [VARIABLES] ObjectNode  objectNode  schema  boolean  
[P5_Replace_Variable]^return schema;^93^^^^^87^94^return objectNode;^[CLASS] JsonSchema  [METHOD] getDefaultSchemaNode [RETURN_TYPE] JsonNode   [VARIABLES] ObjectNode  objectNode  schema  boolean  
