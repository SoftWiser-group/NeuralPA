[P8_Replace_Mix]^this.headerId =  null;^41^^^^^40^42^this.headerId = headerId;^[CLASS] UnrecognizedExtraField  [METHOD] setHeaderId [RETURN_TYPE] void   ZipShort headerId [VARIABLES] byte[]  centralData  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^localData = copy ( localData ) ;^64^^^^^63^65^localData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setLocalFileDataData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P7_Replace_Invocation]^localData = setLocalFileDataData ( data ) ;^64^^^^^63^65^localData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setLocalFileDataData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P8_Replace_Mix]^localData =  copy ( centralData ) ;^64^^^^^63^65^localData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setLocalFileDataData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P11_Insert_Donor_Statement]^centralData = copy ( data ) ;localData = copy ( data ) ;^64^^^^^63^65^localData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setLocalFileDataData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P14_Delete_Statement]^^64^^^^^63^65^localData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setLocalFileDataData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P4_Replace_Constructor]^return return  new ZipShort ( centralData.length )  ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return new ZipShort ( localData ) ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return new ZipShort ( localData.length.length ) ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P8_Replace_Mix]^return  new ZipShort ( centralData.length )  ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return new ZipShort ( data.length ) ;^72^^^^^71^73^return new ZipShort ( localData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return copy ( data ) ;^80^^^^^79^81^return copy ( localData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P7_Replace_Invocation]^return getLocalFileDataData ( localData ) ;^80^^^^^79^81^return copy ( localData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P14_Delete_Statement]^^80^^^^^79^81^return copy ( localData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getLocalFileDataData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^centralData = copy ( localData ) ;^94^^^^^93^95^centralData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setCentralDirectoryData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P7_Replace_Invocation]^centralData = setLocalFileDataData ( data ) ;^94^^^^^93^95^centralData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setCentralDirectoryData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P8_Replace_Mix]^centralData =  copy ( null ) ;^94^^^^^93^95^centralData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setCentralDirectoryData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P11_Insert_Donor_Statement]^localData = copy ( data ) ;centralData = copy ( data ) ;^94^^^^^93^95^centralData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setCentralDirectoryData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P14_Delete_Statement]^^94^^^^^93^95^centralData = copy ( data ) ;^[CLASS] UnrecognizedExtraField  [METHOD] setCentralDirectoryData [RETURN_TYPE] void   byte[] data [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P2_Replace_Operator]^if  ( centralData == null )  {^103^^^^^102^107^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^if  ( localData != null )  {^103^^^^^102^107^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P8_Replace_Mix]^if  ( centralData != false )  {^103^^^^^102^107^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P9_Replace_Statement]^if  ( localData == null )  {^103^^^^^102^107^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P9_Replace_Statement]^if  ( from != null )  {^103^^^^^102^107^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P15_Unwrap_Block]^return new org.apache.commons.compress.archivers.zip.ZipShort(centralData.length);^103^104^105^^^102^107^if  ( centralData != null )  { return new ZipShort ( centralData.length ) ; }^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P16_Remove_Block]^^103^104^105^^^102^107^if  ( centralData != null )  { return new ZipShort ( centralData.length ) ; }^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P13_Insert_Block]^if  (  ( centralData )  != null )  {     return copy ( centralData ) ; }^103^^^^^102^107^[Delete]^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P4_Replace_Constructor]^return return  new ZipShort ( localData.length )  ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return new ZipShort ( localData.length ) ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return new ZipShort ( centralData ) ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return new ZipShort ( centralData.length.length ) ;^104^^^^^102^107^return new ZipShort ( centralData.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P13_Insert_Block]^if  (  ( centralData )  != null )  {     return new ZipShort ( centralData.length ) ; }^104^^^^^102^107^[Delete]^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P7_Replace_Invocation]^return getLocalFileDataData (  ) ;^106^^^^^102^107^return getLocalFileDataLength (  ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P14_Delete_Statement]^^106^^^^^102^107^return getLocalFileDataLength (  ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryLength [RETURN_TYPE] ZipShort   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P2_Replace_Operator]^if  ( centralData == null )  {^114^^^^^113^118^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^if  ( localData != null )  {^114^^^^^113^118^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P8_Replace_Mix]^if  ( centralData != true )  {^114^^^^^113^118^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P9_Replace_Statement]^if  ( localData == null )  {^114^^^^^113^118^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P9_Replace_Statement]^if  ( from != null )  {^114^^^^^113^118^if  ( centralData != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P15_Unwrap_Block]^return org.apache.commons.compress.archivers.zip.UnrecognizedExtraField.copy(centralData);^114^115^116^^^113^118^if  ( centralData != null )  { return copy ( centralData ) ; }^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P16_Remove_Block]^^114^115^116^^^113^118^if  ( centralData != null )  { return copy ( centralData ) ; }^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P13_Insert_Block]^if  (  ( centralData )  != null )  {     return new ZipShort ( centralData.length ) ; }^114^^^^^113^118^[Delete]^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return copy ( localData ) ;^115^^^^^113^118^return copy ( centralData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P7_Replace_Invocation]^return getLocalFileDataData ( centralData ) ;^115^^^^^113^118^return copy ( centralData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P13_Insert_Block]^if  (  ( centralData )  != null )  {     return copy ( centralData ) ; }^115^^^^^113^118^[Delete]^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P14_Delete_Statement]^^115^^^^^113^118^return copy ( centralData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P8_Replace_Mix]^return getLocalFileDataData ( localData ) ;^115^^^^^113^118^return copy ( centralData ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P7_Replace_Invocation]^return getLocalFileDataLength (  ) ;^117^^^^^113^118^return getLocalFileDataData (  ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P14_Delete_Statement]^^117^^^^^113^118^return getLocalFileDataData (  ) ;^[CLASS] UnrecognizedExtraField  [METHOD] getCentralDirectoryData [RETURN_TYPE] byte[]   [VARIABLES] byte[]  centralData  data  localData  ZipShort  headerId  boolean  
[P11_Insert_Donor_Statement]^byte[] to = new byte[from.length];byte[] tmp = new byte[length];^127^^^^^126^130^byte[] tmp = new byte[length];^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( tmp, offset, tmp, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, length, tmp, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset, localData, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset, tmp, 0, offset ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy (  offset, tmp, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data,  tmp, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset,  0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset, tmp, 0 ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( tmp, offset, data, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, tmp, offset, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, length, tmp, 0, offset ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P14_Delete_Statement]^^128^129^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ; setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^System.arraycopy ( from, 0, to, 0, to.length ) ;System.arraycopy ( data, offset, tmp, 0, length ) ;^128^^^^^126^130^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^setLocalFileDataData ( localData ) ;^129^^^^^126^130^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P7_Replace_Invocation]^setCentralDirectoryData ( tmp ) ;^129^^^^^126^130^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P14_Delete_Statement]^^129^^^^^126^130^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^setCentralDirectoryData ( tmp ) ;setLocalFileDataData ( tmp ) ;^129^^^^^126^130^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^return getLocalFileDataLength (  ) ;setLocalFileDataData ( tmp ) ;^129^^^^^126^130^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^return getLocalFileDataData (  ) ;setLocalFileDataData ( tmp ) ;^129^^^^^126^130^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromLocalFileData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^byte[] to = new byte[from.length];byte[] tmp = new byte[length];^140^^^^^138^146^byte[] tmp = new byte[length];^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P3_Replace_Literal]^System.arraycopy ( data, offset, tmp, 6, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( localData, offset, tmp, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, length, tmp, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset, localData, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset, tmp, 0, offset ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy (  offset, tmp, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data,  tmp, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset,  0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset, tmp, 0 ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( offset, data, tmp, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, length, tmp, 0, offset ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^System.arraycopy ( data, offset, length, 0, tmp ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P14_Delete_Statement]^^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^System.arraycopy ( from, 0, to, 0, to.length ) ;System.arraycopy ( data, offset, tmp, 0, length ) ;^141^^^^^138^146^System.arraycopy ( data, offset, tmp, 0, length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^setCentralDirectoryData ( localData ) ;^142^^^^^138^146^setCentralDirectoryData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P7_Replace_Invocation]^setLocalFileDataData ( tmp ) ;^142^^^^^138^146^setCentralDirectoryData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P14_Delete_Statement]^^142^^^^^138^146^setCentralDirectoryData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^setLocalFileDataData ( tmp ) ;setCentralDirectoryData ( tmp ) ;^142^^^^^138^146^setCentralDirectoryData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P2_Replace_Operator]^if  ( localData != null )  {^143^^^^^138^146^if  ( localData == null )  {^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P5_Replace_Variable]^if  ( data == null )  {^143^^^^^138^146^if  ( localData == null )  {^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P8_Replace_Mix]^if  ( localData == true )  {^143^^^^^138^146^if  ( localData == null )  {^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P9_Replace_Statement]^if  ( centralData != null )  {^143^^^^^138^146^if  ( localData == null )  {^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P15_Unwrap_Block]^setLocalFileDataData(tmp);^143^144^145^^^138^146^if  ( localData == null )  { setLocalFileDataData ( tmp ) ; }^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P16_Remove_Block]^^143^144^145^^^138^146^if  ( localData == null )  { setLocalFileDataData ( tmp ) ; }^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P7_Replace_Invocation]^setCentralDirectoryData ( tmp ) ;^144^^^^^138^146^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P8_Replace_Mix]^setLocalFileDataData ( localData ) ;^144^^^^^138^146^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P14_Delete_Statement]^^144^^^^^138^146^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^setCentralDirectoryData ( tmp ) ;setLocalFileDataData ( tmp ) ;^144^^^^^138^146^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^return getLocalFileDataLength (  ) ;setLocalFileDataData ( tmp ) ;^144^^^^^138^146^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P11_Insert_Donor_Statement]^return getLocalFileDataData (  ) ;setLocalFileDataData ( tmp ) ;^144^^^^^138^146^setLocalFileDataData ( tmp ) ;^[CLASS] UnrecognizedExtraField  [METHOD] parseFromCentralDirectoryData [RETURN_TYPE] void   byte[] data int offset int length [VARIABLES] byte[]  centralData  data  localData  tmp  ZipShort  headerId  boolean  int  length  offset  
[P2_Replace_Operator]^if  ( from == null )  {^149^^^^^148^155^if  ( from != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^if  ( tmp != null )  {^149^^^^^148^155^if  ( from != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P8_Replace_Mix]^if  ( tmp != true )  {^149^^^^^148^155^if  ( from != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P9_Replace_Statement]^if  ( centralData != null )  {^149^^^^^148^155^if  ( from != null )  {^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P15_Unwrap_Block]^byte[] to = new byte[from.length]; java.lang.System.arraycopy(from, 0, to, 0, to.length); return to;^149^150^151^152^153^148^155^if  ( from != null )  { byte[] to = new byte[from.length]; System.arraycopy ( from, 0, to, 0, to.length ) ; return to; }^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P16_Remove_Block]^^149^150^151^152^153^148^155^if  ( from != null )  { byte[] to = new byte[from.length]; System.arraycopy ( from, 0, to, 0, to.length ) ; return to; }^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^return tmp;^152^^^^^148^155^return to;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P11_Insert_Donor_Statement]^byte[] tmp = new byte[length];byte[] to = new byte[from.length];^150^^^^^148^155^byte[] to = new byte[from.length];^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P3_Replace_Literal]^System.arraycopy ( from, 5, to, 5, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P3_Replace_Literal]^System.arraycopy ( from, -8, to, -8, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy ( to, 0, to, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy ( from, 0, tmp, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy (  0, to, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy ( from, 0,  0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy ( from, 0, to, 0 ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy ( to, 0, from, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy ( to.length, 0, to, 0, from ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P8_Replace_Mix]^System.arraycopy ( tmp, 0, to, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P14_Delete_Statement]^^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P11_Insert_Donor_Statement]^System.arraycopy ( data, offset, tmp, 0, length ) ;System.arraycopy ( from, 0, to, 0, to.length ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P5_Replace_Variable]^System.arraycopy ( from, 0, to.length, 0, to ) ;^151^^^^^148^155^System.arraycopy ( from, 0, to, 0, to.length ) ;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
[P8_Replace_Mix]^return false;^154^^^^^148^155^return null;^[CLASS] UnrecognizedExtraField  [METHOD] copy [RETURN_TYPE] byte[]   byte[] from [VARIABLES] byte[]  centralData  data  from  localData  tmp  to  ZipShort  headerId  boolean  
