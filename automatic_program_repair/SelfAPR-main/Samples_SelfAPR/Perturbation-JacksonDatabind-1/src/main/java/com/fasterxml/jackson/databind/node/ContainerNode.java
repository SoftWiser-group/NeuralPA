[P5_Replace_Variable]^_nodeFactory = _nodeFactory;^25^^^^^23^26^_nodeFactory = nc;^[CLASS] ContainerNode  [METHOD] <init> [RETURN_TYPE] JsonNodeFactory)   JsonNodeFactory nc [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P8_Replace_Mix]^_nodeFactory =  null;^25^^^^^23^26^_nodeFactory = nc;^[CLASS] ContainerNode  [METHOD] <init> [RETURN_TYPE] JsonNodeFactory)   JsonNodeFactory nc [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P1_Replace_Type]^public char asText (  )  { return ""; }^35^^^^^30^40^public String asText (  )  { return ""; }^[CLASS] ContainerNode  [METHOD] asText [RETURN_TYPE] String   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final ArrayNode arrayNode (  )  { return nc.arrayNode (  ) ; }^63^^^^^58^68^public final ArrayNode arrayNode (  )  { return _nodeFactory.arrayNode (  ) ; }^[CLASS] ContainerNode  [METHOD] arrayNode [RETURN_TYPE] ArrayNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final ArrayNode arrayNode (  )  { return _nodeFactory.nullNode (  ) ; }^63^^^^^58^68^public final ArrayNode arrayNode (  )  { return _nodeFactory.arrayNode (  ) ; }^[CLASS] ContainerNode  [METHOD] arrayNode [RETURN_TYPE] ArrayNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P8_Replace_Mix]^public final ArrayNode arrayNode (  )  { return nc.nullNode (  ) ; }^63^^^^^58^68^public final ArrayNode arrayNode (  )  { return _nodeFactory.arrayNode (  ) ; }^[CLASS] ContainerNode  [METHOD] arrayNode [RETURN_TYPE] ArrayNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final ObjectNode objectNode (  )  { return nc.objectNode (  ) ; }^69^^^^^64^74^public final ObjectNode objectNode (  )  { return _nodeFactory.objectNode (  ) ; }^[CLASS] ContainerNode  [METHOD] objectNode [RETURN_TYPE] ObjectNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final ObjectNode objectNode (  )  { return _nodeFactory.arrayNode (  ) ; }^69^^^^^64^74^public final ObjectNode objectNode (  )  { return _nodeFactory.objectNode (  ) ; }^[CLASS] ContainerNode  [METHOD] objectNode [RETURN_TYPE] ObjectNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P8_Replace_Mix]^public final ObjectNode objectNode (  )  { return nc.arrayNode (  ) ; }^69^^^^^64^74^public final ObjectNode objectNode (  )  { return _nodeFactory.objectNode (  ) ; }^[CLASS] ContainerNode  [METHOD] objectNode [RETURN_TYPE] ObjectNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NullNode nullNode (  )  { return _nodeFactory.arrayNode (  ) ; }^71^^^^^66^76^public final NullNode nullNode (  )  { return _nodeFactory.nullNode (  ) ; }^[CLASS] ContainerNode  [METHOD] nullNode [RETURN_TYPE] NullNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P8_Replace_Mix]^public final NullNode nullNode (  )  { return nc.arrayNode (  ) ; }^71^^^^^66^76^public final NullNode nullNode (  )  { return _nodeFactory.nullNode (  ) ; }^[CLASS] ContainerNode  [METHOD] nullNode [RETURN_TYPE] NullNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NullNode nullNode (  )  { return nc.nullNode (  ) ; }^71^^^^^66^76^public final NullNode nullNode (  )  { return _nodeFactory.nullNode (  ) ; }^[CLASS] ContainerNode  [METHOD] nullNode [RETURN_TYPE] NullNode   [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final BooleanNode booleanNode ( boolean v )  { return nc.booleanNode ( v ) ; }^73^^^^^68^78^public final BooleanNode booleanNode ( boolean v )  { return _nodeFactory.booleanNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] booleanNode [RETURN_TYPE] BooleanNode   boolean v [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  v  
[P5_Replace_Variable]^public final BooleanNode booleanNode ( boolean _nodeFactory )  { return v.booleanNode ( v ) ; }^73^^^^^68^78^public final BooleanNode booleanNode ( boolean v )  { return _nodeFactory.booleanNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] booleanNode [RETURN_TYPE] BooleanNode   boolean v [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  v  
[P7_Replace_Invocation]^public final BooleanNode booleanNode ( boolean v )  { return _nodeFactory.numberNode ( v ) ; }^73^^^^^68^78^public final BooleanNode booleanNode ( boolean v )  { return _nodeFactory.booleanNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] booleanNode [RETURN_TYPE] BooleanNode   boolean v [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  v  
[P8_Replace_Mix]^public final BooleanNode booleanNode ( boolean v )  { return nc .objectNode (  )  ; }^73^^^^^68^78^public final BooleanNode booleanNode ( boolean v )  { return _nodeFactory.booleanNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] booleanNode [RETURN_TYPE] BooleanNode   boolean v [VARIABLES] JsonNodeFactory  _nodeFactory  nc  boolean  v  
[P5_Replace_Variable]^public final NumericNode numberNode ( byte v )  { return nc.numberNode ( v ) ; }^75^^^^^70^80^public final NumericNode numberNode ( byte v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   byte v [VARIABLES] byte  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( byte _nodeFactory )  { return v.numberNode ( v ) ; }^75^^^^^70^80^public final NumericNode numberNode ( byte v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   byte v [VARIABLES] byte  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NumericNode numberNode ( byte v )  { return _nodeFactory .nullNode (  )  ; }^75^^^^^70^80^public final NumericNode numberNode ( byte v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   byte v [VARIABLES] byte  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( short v )  { return nc.numberNode ( v ) ; }^76^^^^^71^81^public final NumericNode numberNode ( short v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   short v [VARIABLES] short  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( short _nodeFactory )  { return v.numberNode ( v ) ; }^76^^^^^71^81^public final NumericNode numberNode ( short v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   short v [VARIABLES] short  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NumericNode numberNode ( short v )  { return _nodeFactory .nullNode (  )  ; }^76^^^^^71^81^public final NumericNode numberNode ( short v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   short v [VARIABLES] short  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( int v )  { return nc.numberNode ( v ) ; }^77^^^^^72^82^public final NumericNode numberNode ( int v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   int v [VARIABLES] int  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( int _nodeFactory )  { return v.numberNode ( v ) ; }^77^^^^^72^82^public final NumericNode numberNode ( int v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   int v [VARIABLES] int  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NumericNode numberNode ( int v )  { return _nodeFactory .nullNode (  )  ; }^77^^^^^72^82^public final NumericNode numberNode ( int v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   int v [VARIABLES] int  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( long v )  { return nc.numberNode ( v ) ; }^78^^^^^73^83^public final NumericNode numberNode ( long v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   long v [VARIABLES] long  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( long _nodeFactory )  { return v.numberNode ( v ) ; }^78^^^^^73^83^public final NumericNode numberNode ( long v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   long v [VARIABLES] long  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( float v )  { return nc.numberNode ( v ) ; }^79^^^^^74^84^public final NumericNode numberNode ( float v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   float v [VARIABLES] float  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( float _nodeFactory )  { return v.numberNode ( v ) ; }^79^^^^^74^84^public final NumericNode numberNode ( float v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   float v [VARIABLES] float  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NumericNode numberNode ( float v )  { return _nodeFactory .nullNode (  )  ; }^79^^^^^74^84^public final NumericNode numberNode ( float v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   float v [VARIABLES] float  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( double v )  { return nc.numberNode ( v ) ; }^80^^^^^75^85^public final NumericNode numberNode ( double v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   double v [VARIABLES] double  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( double _nodeFactory )  { return v.numberNode ( v ) ; }^80^^^^^75^85^public final NumericNode numberNode ( double v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   double v [VARIABLES] double  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NumericNode numberNode ( double v )  { return _nodeFactory .nullNode (  )  ; }^80^^^^^75^85^public final NumericNode numberNode ( double v )  { return _nodeFactory.numberNode ( v ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   double v [VARIABLES] double  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( BigDecimal v )  { return  ( nc.numberNode ( v )  ) ; }^81^^^^^76^86^public final NumericNode numberNode ( BigDecimal v )  { return  ( _nodeFactory.numberNode ( v )  ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   BigDecimal v [VARIABLES] BigDecimal  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final NumericNode numberNode ( BigDecimal _nodeFactory )  { return  ( v.numberNode ( v )  ) ; }^81^^^^^76^86^public final NumericNode numberNode ( BigDecimal v )  { return  ( _nodeFactory.numberNode ( v )  ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   BigDecimal v [VARIABLES] BigDecimal  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NumericNode numberNode ( BigDecimal v )  { return  ( _nodeFactory.POJONode ( v )  ) ; }^81^^^^^76^86^public final NumericNode numberNode ( BigDecimal v )  { return  ( _nodeFactory.numberNode ( v )  ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   BigDecimal v [VARIABLES] BigDecimal  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final NumericNode numberNode ( BigDecimal v )  { return  ( _nodeFactory .nullNode (  )   ) ; }^81^^^^^76^86^public final NumericNode numberNode ( BigDecimal v )  { return  ( _nodeFactory.numberNode ( v )  ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   BigDecimal v [VARIABLES] BigDecimal  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P8_Replace_Mix]^public final NumericNode numberNode ( BigDecimal v )  { return  ( nc.POJONode ( v )  ) ; }^81^^^^^76^86^public final NumericNode numberNode ( BigDecimal v )  { return  ( _nodeFactory.numberNode ( v )  ) ; }^[CLASS] ContainerNode  [METHOD] numberNode [RETURN_TYPE] NumericNode   BigDecimal v [VARIABLES] BigDecimal  v  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final TextNode textNode ( String text )  { return nc.textNode ( text ) ; }^83^^^^^78^88^public final TextNode textNode ( String text )  { return _nodeFactory.textNode ( text ) ; }^[CLASS] ContainerNode  [METHOD] textNode [RETURN_TYPE] TextNode   String text [VARIABLES] JsonNodeFactory  _nodeFactory  nc  String  text  boolean  
[P5_Replace_Variable]^public final TextNode _nodeFactoryNode ( String text )  { return text.textNode ( text ) ; }^83^^^^^78^88^public final TextNode textNode ( String text )  { return _nodeFactory.textNode ( text ) ; }^[CLASS] ContainerNode  [METHOD] textNode [RETURN_TYPE] TextNode   String text [VARIABLES] JsonNodeFactory  _nodeFactory  nc  String  text  boolean  
[P7_Replace_Invocation]^public final TextNode textNode ( String text )  { return _nodeFactory.POJONode ( text ) ; }^83^^^^^78^88^public final TextNode textNode ( String text )  { return _nodeFactory.textNode ( text ) ; }^[CLASS] ContainerNode  [METHOD] textNode [RETURN_TYPE] TextNode   String text [VARIABLES] JsonNodeFactory  _nodeFactory  nc  String  text  boolean  
[P7_Replace_Invocation]^public final TextNode textNode ( String text )  { return _nodeFactory .objectNode (  )  ; }^83^^^^^78^88^public final TextNode textNode ( String text )  { return _nodeFactory.textNode ( text ) ; }^[CLASS] ContainerNode  [METHOD] textNode [RETURN_TYPE] TextNode   String text [VARIABLES] JsonNodeFactory  _nodeFactory  nc  String  text  boolean  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data )  { return nc.binaryNode ( data ) ; }^85^^^^^80^90^public final BinaryNode binaryNode ( byte[] data )  { return _nodeFactory.binaryNode ( data ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data [VARIABLES] byte[]  data  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] _nodeFactory )  { return data.binaryNode ( data ) ; }^85^^^^^80^90^public final BinaryNode binaryNode ( byte[] data )  { return _nodeFactory.binaryNode ( data ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data [VARIABLES] byte[]  data  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final BinaryNode binaryNode ( byte[] data )  { return _nodeFactory.numberNode ( data ) ; }^85^^^^^80^90^public final BinaryNode binaryNode ( byte[] data )  { return _nodeFactory.binaryNode ( data ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data [VARIABLES] byte[]  data  JsonNodeFactory  _nodeFactory  nc  boolean  
[P8_Replace_Mix]^public final BinaryNode binaryNode ( byte[] data )  { return nc.numberNode ( data ) ; }^85^^^^^80^90^public final BinaryNode binaryNode ( byte[] data )  { return _nodeFactory.binaryNode ( data ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data [VARIABLES] byte[]  data  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return nc.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[]  int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data, int  int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] length, int offset, int data )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data, int length, int offset )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data, int _nodeFactory, int length )  { return offset.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P7_Replace_Invocation]^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory .binaryNode ( data )  ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data, int offset, int offset )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] _nodeFactory, int offset, int length )  { return data.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] offset, int data, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final BinaryNode binaryNode ( byte[] data, int offset, int _nodeFactory )  { return length.binaryNode ( data, offset, length ) ; }^86^^^^^81^91^public final BinaryNode binaryNode ( byte[] data, int offset, int length )  { return _nodeFactory.binaryNode ( data, offset, length ) ; }^[CLASS] ContainerNode  [METHOD] binaryNode [RETURN_TYPE] BinaryNode   byte[] data int offset int length [VARIABLES] byte[]  data  boolean  int  length  offset  JsonNodeFactory  _nodeFactory  nc  
[P5_Replace_Variable]^public final POJONode POJONode ( Object _nodeFactory )  { return pojo.POJONode ( pojo ) ; }^88^^^^^83^93^public final POJONode POJONode ( Object pojo )  { return _nodeFactory.POJONode ( pojo ) ; }^[CLASS] ContainerNode  [METHOD] POJONode [RETURN_TYPE] POJONode   Object pojo [VARIABLES] Object  pojo  JsonNodeFactory  _nodeFactory  nc  boolean  
[P7_Replace_Invocation]^public final POJONode POJONode ( Object pojo )  { return _nodeFactory.textNode ( pojo ) ; }^88^^^^^83^93^public final POJONode POJONode ( Object pojo )  { return _nodeFactory.POJONode ( pojo ) ; }^[CLASS] ContainerNode  [METHOD] POJONode [RETURN_TYPE] POJONode   Object pojo [VARIABLES] Object  pojo  JsonNodeFactory  _nodeFactory  nc  boolean  
[P5_Replace_Variable]^public final POJONode POJONode ( Object pojo )  { return nc.POJONode ( pojo ) ; }^88^^^^^83^93^public final POJONode POJONode ( Object pojo )  { return _nodeFactory.POJONode ( pojo ) ; }^[CLASS] ContainerNode  [METHOD] POJONode [RETURN_TYPE] POJONode   Object pojo [VARIABLES] Object  pojo  JsonNodeFactory  _nodeFactory  nc  boolean  
